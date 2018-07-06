use std::env;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    Command::new("nvcc")
        .args(&["src/kernel.cu",
            // static library (.lib) を出力させる
            "-lib",
            // Cコンパイラのパスを指定
            "-ccbin", 
            r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.14.26428\bin\Hostx64\x64\cl.exe",
            // Cコンパイラにwarning 4819を無視するよう指示
            "-Xcompiler", "-wd4819",
            "-o"])
        .arg(&format!("{}/kernel.lib", out_dir))
        .status()
        .unwrap();

    println!("cargo:rustc-link-search=native=C:/PROGRA~1/NVIDIA~2/CUDA/v9.2/lib/x64");
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=kernel");
}
