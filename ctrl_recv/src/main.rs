use std::env;
use std::net::UdpSocket;
use std::process;
use std::str;

#[derive(Debug, Default)]
struct Joints {
    x: f32,
    y: f32,
    z: f32,
    gripper: f32,
}

#[derive(Debug, Clone, Copy)]
struct Limits {
    x: f32,
    y: f32,
    z: f32,
    gripper: f32,
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 4 {
        eprintln!("Usage: ctrl_recv <max_x> <max_y> <max_z> <max_gripper>");
        process::exit(1);
    }

    let parse = |value: &str, name: &str| -> f32 {
        value.parse().unwrap_or_else(|_| {
            eprintln!("Invalid {} value: {}", name, value);
            process::exit(1);
        })
    };

    let limits = Limits {
        x: parse(&args[0], "max_x"),
        y: parse(&args[1], "max_y"),
        z: parse(&args[2], "max_z"),
        gripper: parse(&args[3], "max_gripper"),
    };

    let socket = UdpSocket::bind("0.0.0.0:8080")?;
    let mut buf = [0; 1024];

    loop {
        match socket.recv_from(&mut buf) {
            Ok((amt, _src)) => {
                let filled_buf = &buf[..amt];
                if let Ok(text) = str::from_utf8(filled_buf) {
                    process_data(text, &limits);
                }
            }
            Err(e) => {
                eprintln!("Error receiving data: {}", e);
            }
        }
    }
}

fn process_data(text: &str, limits: &Limits) {
    // <x> <y> <z> <pitch> <yaw> <roll> <wrist_pitch> <wrist_yaw> <wrist_roll>
    let parts: Vec<&str> = text.split_whitespace().collect();

    if parts.len() == 9 {
        let data = Joints {
            x: parts[0].parse().unwrap_or(0f32).clamp(-limits.x, limits.x) / limits.x * 0.85,
            y: parts[1].parse().unwrap_or(0f32).clamp(-limits.y, limits.y) / limits.y * 0.9,
            z: (1f32 - (parts[2].parse().unwrap_or(0f32).clamp(0f32, limits.z) / limits.z)) * 1.1,
            gripper: -parts[6].parse().unwrap_or(0f32).clamp(-limits.gripper, 0f32) / limits.gripper
                * 0.105,
        };

        println!(
            "{:.4} {:.4} {:.4} {:.4}",
            data.x, data.y, data.z, data.gripper
        );
    } else {
        eprintln!("Received malformed data: {}", text);
    }
}
