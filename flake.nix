{
  description = "A fast image viewer. nsxiv meets mpv. JPEG XL first-class.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    crane.url = "github:ipetkov/crane";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    treefmt-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    inputs@{
      flake-parts,
      nixpkgs,
      crane,
      treefmt-nix,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ treefmt-nix.flakeModule ];
      systems = [ "x86_64-linux" ];

      perSystem =
        {
          system,
          pkgs,
          lib,
          config,
          ...
        }:
        let
          craneLib = crane.mkLib pkgs;

          # System libraries raylib-sys compiles against and the final binary
          # links at runtime (X11 / GL family). jxl-oxide is pure Rust, so no
          # libjxl here.
          libInputs = [
            pkgs.libGL
            pkgs.libx11
            pkgs.libxi
            pkgs.libxrandr
            pkgs.libxfixes
            pkgs.libxcursor
            pkgs.libxinerama
          ];

          # Native tools needed by the raylib-sys crate: it compiles the
          # vendored raylib with CMake and generates bindings with bindgen.
          nativeBuildInputs = [
            pkgs.pkg-config
            pkgs.cmake
            pkgs.llvmPackages.libclang
            pkgs.makeWrapper
          ];

          # Env needed by bindgen / clang.
          env = {
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            BINDGEN_EXTRA_CLANG_ARGS = lib.concatStringsSep " " [
              "-isystem"
              "${pkgs.stdenv.cc.libc_dev}/include"
            ];
          };

          # rpath link-args so the binary finds the system libs at runtime.
          rpathFlags = lib.concatStringsSep " " (
            map (p: "-C link-arg=-Wl,-rpath,${p}") (lib.splitString ":" (lib.makeLibraryPath libInputs))
          );

          # Source as seen by cargo: everything tracked by git.
          src = craneLib.cleanCargoSource ./.;

          # Cargo.lock has no network access during the build; fetch deps
          # first, then feed them to cargo through the registry cache.
          cargoArtifacts = craneLib.buildDepsOnly {
            inherit src nativeBuildInputs;
            buildInputs = libInputs;
            inherit env;
          };

          versatileViewer = craneLib.buildPackage {
            inherit
              src
              cargoArtifacts
              env
              ;
            buildInputs = libInputs;
            # raylib dlopens libGL at runtime (it is not in DT_NEEDED), and
            # Nix's fixup step shrinks RUNPATH down to only the DT_NEEDED
            # libs. autoPatchelfHook runs in postFixup (after shrink), so
            # appendRunpaths survives and bakes the dirs into the final
            # DT_RUNPATH, making the binary self-contained. Same idiom as
            # nixpkgs' raylib package.
            nativeBuildInputs = nativeBuildInputs ++ [ pkgs.autoPatchelfHook ];
            appendRunpaths = [ (lib.makeLibraryPath libInputs) ];
            meta.description = "A fast image viewer. nsxiv meets mpv. JPEG XL first-class.";
          };
        in
        {
          packages.versatile-viewer = versatileViewer;
          packages.default = config.packages.versatile-viewer;

          checks.default = craneLib.cargoTest {
            inherit
              src
              cargoArtifacts
              nativeBuildInputs
              env
              ;
            buildInputs = libInputs;
            meta.description = "Versatile-viewer test suite";
          };

          apps.default = {
            type = "app";
            program = "${versatileViewer}/bin/vv";
            meta.description = "A fast image (and later, video) viewer. JPEG XL first-class.";
            meta.license = lib.licenses.agpl3Plus;
          };

          treefmt = {
            programs.rustfmt.enable = true;
            programs.rustfmt.edition = "2024";
            programs.taplo.enable = true;
            programs.nixfmt.enable = true;
            settings.formatter.rustfmt.options = lib.mkAfter [
              "--config"
              "max_width=100,comment_width=100,wrap_comments=false,group_imports=StdExternalCrate,imports_granularity=Crate,condense_wildcard_suffixes=true,format_code_in_doc_comments=true,format_macro_matchers=true,format_macro_bodies=true,format_strings=true,use_field_init_shorthand=true"
            ];
          };

          devShells.default = craneLib.devShell {
            packages = [
              config.treefmt.build.wrapper
            ]
            ++ nativeBuildInputs;
            buildInputs = libInputs;
            inherit env;
            LD_LIBRARY_PATH = lib.makeLibraryPath libInputs;
            RUSTFLAGS = rpathFlags;
          };
        };
    };
}
