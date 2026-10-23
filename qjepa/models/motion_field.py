"""p47 (phase2.motion_field): the camera's path during the exposure, read off the noisy gyro, as the restorer's input.

The corruption blurs a frame by the camera's own rotation over the exposure (corruptions/motion.py,
``motion_from_imu``). The noisy IMU window the model is given measures that very rotation, so it says which way the
frame was smeared and along what curve -- the one thing an image alone cannot tell for sure. Measured on p22's
noise with a Wiener filter: a kernel built from the noisy gyro took IMU-blurred frames 26.0 -> 33.7 dB when the
exposure was known, 32.2 dB over 5 candidate exposures (README, p24 diagnosis).

Here the path is integrated the way motion.exposure_path does: pitch -> vertical, yaw -> horizontal (TartanAir's
lcam_front is the IMU body frame), roll left out as the blur leaves it out. The exposure is not known at inference,
so the path covers the longest one the corruption draws (``seconds``) around the frame time; a shorter exposure is
its middle part, and darker frames get longer exposures (exposure_tracks_darkness), which the restorer can see.
``samples`` points (odd, the middle one the frame time) give 2 * (samples - 1) numbers -- each point's offset from
the middle in pixels, over ``scale_px`` -- and one more, 1 when an IMU is there at all (an all-zero window, the
p46 "no IMU" sample, reads no gravity). The same numbers go to every pixel as constant input channels, as SRMD
(Zhang et al., CVPR 2018) hands its network the blur kernel and DeepGyro (Mustaniemi et al., WACV 2019) the
gyro's blur field.

The focal length in pixels is ``focal_px`` for a frame ``reference_width`` wide and scales with the frame's width:
the same rotation covers twice the pixels of a frame twice as wide (TartanAir: 90 degrees across, f = width / 2).
Training crops come at many scales (data.source_scale_range), so the true focal length of a crop is not known
either; this one is a middle value and the network learns how far to trust the length. No parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# [ax, ay, az, gx, gy, gz]; motion.py: gx roll, gy pitch (vertical), gz yaw (horizontal).
PITCH_ROW, YAW_ROW = 4, 5
# A real accelerometer reads gravity, 9.8 m/s^2: below this mean norm the window is the all-zero "no IMU" one.
PRESENT_ACCEL = 1.0


class GyroBlurField(nn.Module):
    def __init__(self, seconds: float, samples: int, focal_px: float, scale_px: float, max_px: float,
                 reference_width: int = 256) -> None:
        super().__init__()
        if samples < 3 or samples % 2 == 0:
            raise ValueError("motion_field.samples must be odd and at least 3")
        if min(seconds, focal_px, scale_px, max_px) <= 0:
            raise ValueError("motion_field seconds, focal_px, scale_px and max_px must be positive")
        self.seconds, self.samples = float(seconds), int(samples)
        self.focal_px, self.scale_px, self.max_px = float(focal_px), float(scale_px), float(max_px)
        self.reference_width = int(reference_width)

    @property
    def channels(self) -> int:
        return 2 * (self.samples - 1) + 1

    def path(self, imu_phys: torch.Tensor, imu_times: torch.Tensor, image_time: torch.Tensor,
             width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(u [B,K], v [B,K], present [B]): the principal point's offset in pixels from mid-exposure at the K sample
        times, u horizontal and v vertical as in motion.exposure_path, and whether there is an IMU."""
        with torch.autocast(device_type=imu_phys.device.type, enabled=False):
            imu = imu_phys.float()
            # Relative times in float64 first: absolute timestamps lose the 100 Hz spacing in float32.
            relative = (imu_times.double() - image_time.double()[:, None]).float().contiguous()
            half = 0.5 * self.seconds
            query = torch.linspace(-half, half, self.samples, device=imu.device, dtype=torch.float32)
            query = query.expand(imu.shape[0], -1).contiguous()
            after = torch.searchsorted(relative, query).clamp(1, relative.shape[1] - 1)
            before = after - 1
            t0, t1 = relative.gather(1, before), relative.gather(1, after)
            weight = ((query - t0) / (t1 - t0).clamp_min(1e-6)).clamp(0.0, 1.0)

            def at(row: int) -> torch.Tensor:
                signal = imu[:, row]
                return signal.gather(1, before) + weight * (signal.gather(1, after) - signal.gather(1, before))

            step = self.seconds / (self.samples - 1)

            def angle(rate: torch.Tensor) -> torch.Tensor:
                increments = 0.5 * (rate[:, 1:] + rate[:, :-1]) * step
                total = torch.cat((rate.new_zeros(rate.shape[0], 1), increments.cumsum(dim=1)), dim=1)
                return total - total[:, self.samples // 2:self.samples // 2 + 1]

            focal = self.focal_px * float(width) / self.reference_width
            u = -focal * angle(at(YAW_ROW))
            v = focal * angle(at(PITCH_ROW))
            present = (imu[:, :3].norm(dim=1).mean(dim=1) > PRESENT_ACCEL).float()
            return u * present[:, None], v * present[:, None], present

    def forward(self, imu_phys: torch.Tensor, imu_times: torch.Tensor, image_time: torch.Tensor,
                width: int) -> torch.Tensor:
        """[B, channels]: the path's offsets (middle point dropped, it is 0 by construction) over scale_px, clipped
        at max_px, then the IMU-present flag."""
        u, v, present = self.path(imu_phys, imu_times, image_time, width)
        keep = [k for k in range(self.samples) if k != self.samples // 2]
        limit = self.max_px / self.scale_px
        offsets = (torch.cat((u[:, keep], v[:, keep]), dim=1) / self.scale_px).clamp(-limit, limit)
        return torch.cat((offsets, present[:, None]), dim=1)
