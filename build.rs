fn main() {
    // Specify the path to where OpenBLAS is installed
    println!("cargo:rustc-link-arg=-L/path/to/openblas/lib");
    // Link against the OpenBLAS library
    println!("cargo:rustc-link-arg=-lopenblas");
}
