use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest_path = Path::new(&manifest_dir);

    let mut link_dir: Option<PathBuf> = None;

    if let Ok(dir) = env::var("POMAIDB_LIB_DIR") {
        let p = PathBuf::from(dir);
        if p.exists() {
            link_dir = Some(p);
        }
    }

    if link_dir.is_none() {
        let local_lib = manifest_path.join("lib");
        if local_lib.exists() {
            link_dir = Some(local_lib);
        }
    }

    if link_dir.is_none() {
        let repo_build = manifest_path.join("../../build");
        if repo_build.exists() {
            link_dir = Some(repo_build);
        }
    }

    if let Some(ref dir) = link_dir {
        println!("cargo:rustc-link-search=native={}", dir.display());

        // On Windows, copy dll to OUT_DIR and target dir for cargo test / run
        if cfg!(windows) {
            let dll_name = "libpomai_c.dll";
            let src_dll = dir.join(dll_name);
            if src_dll.exists() {
                if let Ok(out_dir) = env::var("OUT_DIR") {
                    let out_path = PathBuf::from(&out_dir);
                    let _ = fs::copy(&src_dll, out_path.join(dll_name));
                    // Also try copy to target/<profile>/
                    if let Some(target_dir) = out_path.ancestors().nth(3) {
                        let _ = fs::copy(&src_dll, target_dir.join(dll_name));
                        let deps_dir = target_dir.join("deps");
                        if deps_dir.exists() {
                            let _ = fs::copy(&src_dll, deps_dir.join(dll_name));
                        }
                    }
                }
            }
        }
        if !cfg!(windows) {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
    }

    println!("cargo:rustc-link-lib=pomai_c");
    println!("cargo:rerun-if-env-changed=POMAIDB_LIB_DIR");
    println!("cargo:rerun-if-changed=build.rs");
}