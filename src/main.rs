extern crate image;

mod cuda_runtime;
mod cuda_wrap;

use std::os::raw::{c_int, c_void};

extern "C" {
    fn vectorAdd(a: *const f32, b: *const f32, c: *mut f32, n: c_int) -> c_void;
    fn initImage(data: *mut u8, cols: c_int, rows: c_int) -> c_void;
}

fn main() {
    const W: usize = 800;
    const H: usize = 600;
    let mut data = vec![0u8; 3 * W * H];

    let mut dev_data: *mut u8 = cuda_wrap::malloc(3 * W * H).unwrap();

    let blockdim = cuda_runtime::dim3 { x: 32, y: 32, z: 1 };
    let griddim = cuda_runtime::dim3 {
        x: ((W + 31) / 32) as u32,
        y: ((H + 31) / 32) as u32,
        z: 1,
    };
    let sharedmem = 0usize;
    let w_int = W as c_int;
    let h_int = H as c_int;

    cuda_wrap::launch(
        initImage as *const c_void,
        griddim,
        blockdim,
        &mut [
            &mut dev_data as *mut *mut u8 as *mut c_void,
            &w_int as *const c_int as *mut c_void,
            &h_int as *const c_int as *mut c_void,
        ],
        sharedmem,
    ).unwrap();

    cuda_wrap::memcpy(
        data.as_mut_ptr(),
        dev_data,
        3 * W * H,
        cuda_runtime::cudaMemcpyKind_cudaMemcpyDeviceToHost,
    ).unwrap();

    let img = image::ImageBuffer::<image::Rgb<u8>, _>::from_vec(W as u32, H as u32, data).unwrap();
    img.save("sample.png").unwrap();

    //------------------

    const N: usize = 10;

    let vec_a = (0..N + 1).map(|n| n as f32).collect::<Vec<f32>>();
    let vec_b = vec![-1f32; N];
    let mut vec_c = vec![0.0f32; N];

    let mut dev_a: *mut f32 = cuda_wrap::malloc(N * 4).unwrap();
    let mut dev_b: *mut f32 = cuda_wrap::malloc(N * 4).unwrap();
    let mut dev_c: *mut f32 = cuda_wrap::malloc(N * 4).unwrap();

    cuda_wrap::memcpy(
        dev_a,
        vec_a.as_ptr(),
        N * 4,
        cuda_runtime::cudaMemcpyKind_cudaMemcpyHostToDevice,
    ).unwrap();
    cuda_wrap::memcpy(
        dev_b,
        vec_b.as_ptr(),
        N * 4,
        cuda_runtime::cudaMemcpyKind_cudaMemcpyHostToDevice,
    ).unwrap();

    let threads_per_block = 256usize;
    let blockdim = cuda_wrap::usize_to_dim3(threads_per_block);
    let griddim = cuda_wrap::usize_to_dim3((N + threads_per_block - 1) / threads_per_block);
    let sharedmem = 0usize;
    let n_int = N as c_int;

    cuda_wrap::launch(
        vectorAdd as *const c_void,
        griddim,
        blockdim,
        &mut [
            &mut dev_a as *mut *mut f32 as *mut c_void,
            &mut dev_b as *mut *mut f32 as *mut c_void,
            &mut dev_c as *mut *mut f32 as *mut c_void,
            &n_int as *const c_int as *mut c_void,
        ],
        sharedmem,
    ).unwrap();

    cuda_wrap::memcpy(
        vec_c.as_mut_ptr(),
        dev_c,
        N * 4,
        cuda_runtime::cudaMemcpyKind_cudaMemcpyDeviceToHost,
    ).unwrap();

    println!("{:?}", vec_c);
}
