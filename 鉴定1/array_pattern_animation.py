import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


C = 299_792_458  # Speed of light (m/s)
FLOOR_DB = -40.0


def array_factor(theta: np.ndarray, freq_hz: float, n_elements: int, spacing: float, rot_angle: float) -> np.ndarray:
    """Return normalized array-factor power pattern of a rotated uniform linear array."""
    wavelength = C / freq_hz
    k = 2 * np.pi / wavelength

    # Element positions on array axis, centered at origin.
    idx = np.arange(n_elements) - (n_elements - 1) / 2
    axis = np.array([np.cos(rot_angle), np.sin(rot_angle)])
    positions = (idx * spacing)[:, None] * axis[None, :]

    # Observation direction unit vectors on x-y plane.
    u = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    phase = k * positions @ u.T
    af = np.sum(np.exp(1j * phase), axis=0)

    power = np.abs(af) ** 2
    return power / power.max()


def to_db(power: np.ndarray, floor_db: float = FLOOR_DB) -> np.ndarray:
    min_power = 10 ** (floor_db / 10)
    return 10 * np.log10(np.maximum(power, min_power))


def keep_front_half(theta: np.ndarray, power: np.ndarray, rot_angle: float) -> np.ndarray:
    """
    Keep only front hemisphere and force back hemisphere to zero.
    Front direction is broadside (+90° from array axis).
    """
    broadside = rot_angle + np.pi / 2
    delta = np.angle(np.exp(1j * (theta - broadside)))  # wrapped to [-pi, pi]
    masked = power.copy()
    masked[np.abs(delta) > (np.pi / 2)] = 0.0
    return masked


def save_static_pattern(angle_deg: np.ndarray, values: np.ndarray, ylabel: str, title: str, save_path: str) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(angle_deg, values, lw=2)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 360)
    ax.grid(True, alpha=0.35)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_rotation_gif(
    angle_deg: np.ndarray,
    theta: np.ndarray,
    freq_hz: float,
    n_elements: int,
    spacing: float,
    frames: int,
    fps: int,
    db_mode: bool,
    spin_rps: float,
    save_path: str,
) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, 360)
    ax.set_xlabel("Angle (deg)")
    ax.grid(True, alpha=0.35)

    if db_mode:
        ax.set_ylim(FLOOR_DB, 0.5)
        ax.set_ylabel("Normalized Power (dB)")
    else:
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Normalized Power")

    def update(frame: int):
        rot = 2 * np.pi * spin_rps * (frame / fps)
        pat = array_factor(theta, freq_hz=freq_hz, n_elements=n_elements, spacing=spacing, rot_angle=rot)
        pat = keep_front_half(theta, pat, rot_angle=rot)
        y = to_db(pat) if db_mode else pat
        line.set_data(angle_deg, y)
        mode = "dB" if db_mode else "linear"
        ax.set_title(
            f"24 GHz, 12-Element ULA ({mode}, back=0)  |  rotation = {(np.degrees(rot) % 360):.1f}°"
        )

        return (line,)

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    fig.tight_layout()
    ani.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_rotation_gif_polar(
    angle_deg: np.ndarray,
    theta: np.ndarray,
    freq_hz: float,
    n_elements: int,
    spacing: float,
    frames: int,
    fps: int,
    db_mode: bool,
    spin_rps: float,
    save_path: str,
) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="polar")
    line, = ax.plot([], [], lw=2)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.grid(True, alpha=0.35)

    if db_mode:
        ax.set_ylim(FLOOR_DB, 0.5)
        ax.set_yticks([-40, -30, -20, -10, 0])
    else:
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

    polar_angle = np.deg2rad(angle_deg)

    def update(frame: int):
        rot = 2 * np.pi * spin_rps * (frame / fps)
        pat = array_factor(theta, freq_hz=freq_hz, n_elements=n_elements, spacing=spacing, rot_angle=rot)
        pat = keep_front_half(theta, pat, rot_angle=rot)
        y = to_db(pat) if db_mode else pat
        line.set_data(polar_angle, y)
        mode = "dB" if db_mode else "linear"
        ax.set_title(
            f"24 GHz, 12-Element ULA Polar ({mode}, back=0)  |  rotation = {(np.degrees(rot) % 360):.1f}°",
            pad=18,
        )
        return (line,)

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    fig.tight_layout()
    ani.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    freq_hz = 24e9
    n_elements = 12
    wavelength = C / freq_hz
    spacing = wavelength / 2

    angle_deg = np.linspace(0, 360, 1440, endpoint=False)
    theta = np.deg2rad(angle_deg)

    static_pattern = array_factor(theta, freq_hz, n_elements, spacing, rot_angle=0.0)
    static_pattern = keep_front_half(theta, static_pattern, rot_angle=0.0)
    save_static_pattern(
        angle_deg=angle_deg,
        values=static_pattern,
        ylabel="Normalized Power",
        title="24 GHz, 12-Element ULA Direction Pattern (0°~360°, back=0)",
        save_path="direction_pattern_24GHz_12elements_360_back0.png",
    )
    save_static_pattern(
        angle_deg=angle_deg,
        values=to_db(static_pattern),
        ylabel="Normalized Power (dB)",
        title="24 GHz, 12-Element ULA Direction Pattern in dB (0°~360°, back=0)",
        save_path="direction_pattern_24GHz_12elements_360_back0_db.png",
    )

    save_rotation_gif(
        angle_deg=angle_deg,
        theta=theta,
        freq_hz=freq_hz,
        n_elements=n_elements,
        spacing=spacing,
        frames=120,
        fps=20,
        db_mode=False,
        spin_rps=1.0,
        save_path="direction_pattern_rotation_360_back0.gif",
    )
    save_rotation_gif(
        angle_deg=angle_deg,
        theta=theta,
        freq_hz=freq_hz,
        n_elements=n_elements,
        spacing=spacing,
        frames=120,
        fps=20,
        db_mode=True,
        spin_rps=1.0,
        save_path="direction_pattern_rotation_360_back0_db.gif",
    )
    save_rotation_gif_polar(
        angle_deg=angle_deg,
        theta=theta,
        freq_hz=freq_hz,
        n_elements=n_elements,
        spacing=spacing,
        frames=120,
        fps=20,
        db_mode=False,
        spin_rps=1.0,
        save_path="direction_pattern_rotation_polar_360_back0.gif",
    )
    save_rotation_gif_polar(
        angle_deg=angle_deg,
        theta=theta,
        freq_hz=freq_hz,
        n_elements=n_elements,
        spacing=spacing,
        frames=270,
        fps=30,
        db_mode=True,
        spin_rps=1.0,
        save_path="direction_pattern_rotation_polar_360_back0_db.gif",
    )

    print("Done:")
    print("- direction_pattern_24GHz_12elements_360_back0.png")
    print("- direction_pattern_24GHz_12elements_360_back0_db.png")
    print("- direction_pattern_rotation_360_back0.gif")
    print("- direction_pattern_rotation_360_back0_db.gif")
    print("- direction_pattern_rotation_polar_360_back0.gif")
    print("- direction_pattern_rotation_polar_360_back0_db.gif")


if __name__ == "__main__":
    main()
