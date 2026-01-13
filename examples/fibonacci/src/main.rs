use jolt_sdk::{
    host::Program, JoltSharedPreprocessing, JoltProverPreprocessing,
    JoltVerifierPreprocessing, RV64IMACProver, MemoryConfig, MemoryLayout, F, PCS, RV64IMACVerifier, JoltDevice,
    serialize_and_print_size, CommitmentScheme,
};
use std::time::Instant;
use std::path::PathBuf;
use tracing::info;

pub fn main() {
    let _ = tracing_subscriber::fmt().try_init();

    let save_to_disk = std::env::args().any(|arg| arg == "--save");

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = Program::new("fibonacci-guest");
    program.set_func("fib");
    program.build_with_channel(target_dir, "stable");

    let (bytecode, memory_init, program_size) = program.decode();
    println!("bytecode len is {}", bytecode.len());
    let memory_config = MemoryConfig {
        max_input_size: 4096,
        max_output_size: 4096,
        max_untrusted_advice_size: 4096,
        max_trusted_advice_size: 4096,
        stack_size: 4096,
        memory_size: 32768,
        program_size: Some(program_size),
    };
    let memory_layout = MemoryLayout::new(&memory_config);
    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode,
        memory_layout,
        memory_init,
        65536,
    );

    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let verifier_setup = PCS::setup_verifier(&prover_preprocessing.generators);
    let verifier_preprocessing =
        JoltVerifierPreprocessing::new(shared_preprocessing, verifier_setup);

    if save_to_disk {
        serialize_and_print_size(
            "Verifier Preprocessing",
            "/tmp/jolt_verifier_preprocessing.dat",
            &verifier_preprocessing,
        )
            .expect("Could not serialize preprocessing.");
    }

    // Analyze (n=10)
    // Create a separate program instance for analysis because trace_analyze consumes it.
    let mut program_analyze = Program::new("fibonacci-guest");
    program_analyze.set_func("fib");
    program_analyze.build_with_channel(target_dir, "stable");

    let n_arg: u32 = 10;
    let mut input_bytes = vec![];
    input_bytes.append(&mut jolt_sdk::postcard::to_stdvec(&n_arg).unwrap());

    let program_summary = program_analyze.trace_analyze::<F>(&input_bytes, &[], &[]);
    program_summary
        .write_to_file("fib_10.txt".into())
        .expect("should write");

    let trace_file = PathBuf::from("/tmp/fib_trace.bin");
    // Trace to file (n=50)
    let n_arg: u32 = 50;
    let mut input_bytes = vec![];
    input_bytes.append(&mut jolt_sdk::postcard::to_stdvec(&n_arg).unwrap());
    program.trace_to_file(&input_bytes, &[], &[], &trace_file);
    info!("Trace file written to: {:?}.", trace_file);

    let now = Instant::now();
    // Prove (n=50)
    let elf_contents = program.get_elf_contents().expect("elf");
    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &input_bytes,
        &[],
        &[],
        None, None
    );
    let io_device = prover.program_io.clone();
    let (proof, _) = prover.prove();

    // Decode output
    let mut outputs = io_device.outputs.clone();
    outputs.resize(4096, 0); // max_output_size
    let output: u128 = jolt_sdk::postcard::from_bytes(&outputs).expect("deserialize output");

    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    if save_to_disk {
        serialize_and_print_size("Proof", "/tmp/fib_proof.bin", &proof)
            .expect("Could not serialize proof.");
        serialize_and_print_size("io_device", "/tmp/fib_io_device.bin", &io_device)
            .expect("Could not serialize io_device.");
    }

    // Verify
    let mut program_io = JoltDevice::new(&memory_config);
    program_io.inputs = input_bytes.clone();
    program_io.outputs = jolt_sdk::postcard::to_stdvec(&output).unwrap();
    program_io.panic = io_device.panic;

    let is_valid = match RV64IMACVerifier::new(&verifier_preprocessing, proof, program_io, None, None) {
        Ok(verifier) => verifier.verify().is_ok(),
        Err(e) => {
            info!("Verifier creation failed: {:?}", e);
            false
        }
    };

    info!("output: {output}");
    info!("valid: {is_valid}");
}
pub fn main1() {
    // 初始化日志系统，这对于查看 execution trace 和性能统计信息非常重要。
    let _ = tracing_subscriber::fmt().try_init();

    // 检查命令行参数，看是否提供了 "--save" 标志。
    // 如果提供，程序会将预处理数据、证明和 IO 设备状态序列化并保存到磁盘。
    let save_to_disk = std::env::args().any(|arg| arg == "--save");

    // 指定客户程序（Guest Program）编译产物的存储目录。
    let target_dir = "/tmp/jolt-guest-targets";
    // 编译 Fibonacci 客户程序。这将生成 ELF 文件并为后续的预处理做准备。
    let mut program = guest::compile_fib(target_dir);

    // 执行共享预处理步骤。
    // 这一步生成的数据是证明者（Prover）和验证者（Verifier）都需要的公共参数。
    let shared_preprocessing = guest::preprocess_shared_fib(&mut program);

    // 执行证明者预处理。
    // 使用共享预处理数据来生成证明者特定的密钥和参数。
    let prover_preprocessing = guest::preprocess_prover_fib(shared_preprocessing.clone());

    // 生成验证者所需的设置（Setup）。
    // 从证明者预处理生成的生成器（Generators）中提取验证所需的参数。
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();

    // 执行验证者预处理。
    // 使用共享预处理数据和之前提取的设置，生成验证者用于验证证明的结构。
    let verifier_preprocessing =
        guest::preprocess_verifier_fib(shared_preprocessing, verifier_setup);

    // 如果设置了保存到磁盘，则将验证者预处理数据序列化写入文件。
    // 这对于需要在不同机器或时间点进行验证的场景很有用。
    if save_to_disk {
        serialize_and_print_size(
            "Verifier Preprocessing",
            "/tmp/jolt_verifier_preprocessing.dat",
            &verifier_preprocessing,
        )
            .expect("Could not serialize preprocessing.");
    }

    // 构建证明生成闭包/函数。
    // 这个函数封装了程序本身和证明者预处理数据，随后可以被调用来生成具体输入的证明。
    let prove_fib = guest::build_prover_fib(program, prover_preprocessing);

    // 构建验证闭包/函数。
    // 这个函数封装了验证者预处理数据。
    let verify_fib = guest::build_verifier_fib(verifier_preprocessing);

    // 分析 Fibonacci 程序在输入为 10 时的资源消耗（如指令数）。
    let program_summary = guest::analyze_fib(10);
    // 将分析概要写入文本文件。
    program_summary
        .write_to_file("fib_10.txt".into())
        .expect("should write");

    // 生成程序的执行轨迹（Trace）并写入二进制文件。
    // 这对调试生成的证明系统因为 trace 包含了每一步执行的状态。
    let trace_file = "/tmp/fib_trace.bin";
    guest::trace_fib_to_file(trace_file, 50);
    info!("Trace file written to: {trace_file}.");

    // 记录开始时间以测量证明生成耗时。
    let now = Instant::now();
    // 运行证明者生成证明。
    // 输入参数为 50。返回计算结果 (output)、加密证明 (proof) 和 IO 设备状态 (io_device)。
    let (output, proof, io_device) = prove_fib(50);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    // 如果设置了保存到磁盘，将证明和 IO 状态序列化并保存。
    if save_to_disk {
        serialize_and_print_size("Proof", "/tmp/fib_proof.bin", &proof)
            .expect("Could not serialize proof.");
        serialize_and_print_size("io_device", "/tmp/fib_io_device.bin", &io_device)
            .expect("Could not serialize io_device.");
    }

    // 运行验证者函数验证证明的有效性。
    // 需要传入输入参数(50)、计算输出、恐慌状态（是否有 panic）以及证明对象。
    let is_valid = verify_fib(50, output, io_device.panic, proof);
    info!("output: {output}");
    info!("valid: {is_valid}");
}