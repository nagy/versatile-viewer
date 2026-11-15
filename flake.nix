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

          # The system libraries the binary loads at runtime.
          #
          # Every decoder is pure Rust, so nothing compiles C here.
          # winit opens the X11 libraries with `dlopen` as well, and wgpu
          # opens `libvulkan.so.1` the same way, so the two families stay
          # dependencies of the build and of the rpath.
          # The Vulkan driver itself is the host's: the loader reads the ICD
          # manifests of the machine (/run/opengl-driver on NixOS).
          libInputs = [
            pkgs.libx11
            pkgs.libxcb
            pkgs.libxkbcommon
            pkgs.libxi
            pkgs.libxrandr
            pkgs.libxfixes
            pkgs.libxcursor
            pkgs.vulkan-loader
          ];

          # The native tools of the build.
          # No CMake, no clang and no pkg-config: nothing here compiles C.
          # mold links the viewer binary and the ten test binaries of the
          # workspace.
          # Every worker builds in parallel and the linker runs alone at the end
          # of each crate, so the linker is the tail of the build and mold
          # shortens it.
          nativeBuildInputs = [
            pkgs.makeWrapper
            pkgs.mold
          ];

          # The environment the build reads.
          # `RUSTFLAGS` names the linker for every cargo invocation of the
          # flake, so the package drv and the check drvs agree on one flag set
          # and reuse one artifact set.
          env = {
            RUSTFLAGS = "-C link-arg=-fuse-ld=mold";
          };

          # rpath link arguments, so the binary finds the system libraries at
          # runtime.
          rpathFlags = lib.concatStringsSep " " (
            map (p: "-C link-arg=-Wl,-rpath,${p}") (lib.splitString ":" (lib.makeLibraryPath libInputs))
          );

          # The root manifest holds no package, so crane cannot infer a name and
          # a version from it: `buildPackage`, the checks and the docs all take
          # these two.
          # The version is read from `[workspace.package]`, so a bump moves one
          # file and no Nix line.
          # The project name, so a store path matches the flake attribute that
          # carries it (`packages.versatile-viewer`).
          pname = "versatile-viewer";
          version = (builtins.fromTOML (builtins.readFile ./Cargo.toml)).workspace.package.version;

          # The profile of the fast gates: the tests, the headless paints and
          # clippy.
          # crane reads `CARGO_PROFILE` in `cargoWithProfile`, and it answers
          # `release` when the variable stays unset, so every check of this flake
          # built with `lto = "thin"` and `codegen-units = 2` before this table
          # existed.
          # A test binary gains nothing from link-time optimization: the suite
          # runs about 280 tests in 5 s either way, and the release build of the
          # ten binaries cost 56 minutes of CPU and 3 minutes of wall clock for
          # that difference alone.
          # `debug = 1` writes line tables only, so a failing test still names its
          # line and the build skips the full debuginfo.
          devEnv = {
            CARGO_PROFILE = "dev";
            CARGO_PROFILE_DEV_DEBUG = "1";
            CARGO_PROFILE_TEST_DEBUG = "1";
          };

          # The source as cargo sees it.
          # This is everything tracked by git.
          src = craneLib.cleanCargoSource ./.;

          # Cargo.lock has no network access during the build.
          # Fetch the dependencies first.
          # Then feed them to cargo through the registry cache.
          # `--workspace` is the selection every consumer below passes as well:
          # cargo fingerprints the dependencies per invocation, so a drifted
          # selection would recompile the whole graph inside the package drv.
          cargoArtifacts = craneLib.buildDepsOnly {
            inherit
              src
              nativeBuildInputs
              pname
              version
              ;
            buildInputs = libInputs;
            inherit env;
            cargoExtraArgs = "--workspace --locked";
          };

          # The same dependency cache in the profile of the fast gates.
          # cargo keys every artifact on its profile, so a dev-profile check
          # cannot read a release-profile cache and would compile the whole graph
          # again on its own.
          # The package keeps the release cache above, because it ships the
          # optimized binary.
          cargoArtifactsDev = craneLib.buildDepsOnly {
            inherit
              src
              nativeBuildInputs
              pname
              version
              ;
            buildInputs = libInputs;
            env = env // devEnv;
            cargoExtraArgs = "--workspace --locked";
          };

          versatileViewer = craneLib.buildPackage {
            inherit
              src
              cargoArtifacts
              env
              pname
              version
              ;
            # The workspace builds every member, and `vv` is the one binary it
            # installs.
            cargoExtraArgs = "--workspace --locked";
            # `libgcc_s` is a real DT_NEEDED dependency of the binary (unwinding),
            # so the patchelf step needs it at build time.
            buildInputs = libInputs ++ [ pkgs.stdenv.cc.cc.lib ];
            # wgpu opens `libvulkan.so.1` and winit opens the X11 libraries at
            # runtime, so none of them sits in DT_NEEDED.
            # Nix's fixup step shrinks RUNPATH down to only the DT_NEEDED
            # libraries.
            # autoPatchelfHook runs in postFixup, after shrink.
            # appendRunpaths thus survives and bakes the directories into the
            # final DT_RUNPATH.
            # The binary is then self-contained, and the Vulkan driver still
            # comes from the host.
            nativeBuildInputs = nativeBuildInputs ++ [ pkgs.autoPatchelfHook ];
            appendRunpaths = [ (lib.makeLibraryPath libInputs) ];
            # `cargo test` in this derivation repeats the whole test suite of
            # `checks.default`, so it buys no gate at the price of one more
            # release build of the crate and every test binary.
            # The rpath and the missing-library gate live in the fixup, which
            # still runs.
            doCheck = false;
            meta.description = "A fast image viewer. nsxiv meets mpv. JPEG XL first-class.";
          };

          # Static rustdoc HTML installed under $out/share/doc.
          # This is what `cargo doc` writes to ./target/doc/.
          # It reuses the same cargoArtifacts, so only doc crates compile.
          # --no-deps (crane's default) keeps third-party crates out of the
          # search index.
          # Browse offline:
          # nix run nixpkgs#python3 -- -m http.server -d <doc-out>/share/doc
          docs = craneLib.cargoDoc {
            inherit
              src
              cargoArtifacts
              nativeBuildInputs
              env
              pname
              version
              ;
            buildInputs = libInputs;
            # One HTML tree per member, so the whole workspace lands in the
            # store.
            cargoExtraArgs = "--workspace --locked";
            meta.description = "Versatile-viewer API documentation";
          };

          # Lint gate: warnings are errors and every target is included.
          # This uses the same clippy and cargo from nixpkgs as the devshell.
          # `nix flake check` and a plain `cargo clippy` then agree.
          # They do not drift with the toolchain on the developer's PATH.
          # `all = "deny"` in Cargo.toml only applies under clippy.
          # Without this check the build, test and cargoDoc paths never see a
          # lint error at all.
          clippy = craneLib.cargoClippy {
            inherit
              src
              nativeBuildInputs
              pname
              version
              ;
            cargoArtifacts = cargoArtifactsDev;
            env = env // devEnv;
            buildInputs = libInputs;
            # Every member is linted, not only the binary crate.
            cargoExtraArgs = "--workspace --locked";
            cargoClippyExtraArgs = "--all-targets -- --deny warnings";
            meta.description = "Versatile-viewer clippy lint gate";
          };

          # `cargo test` runs the unit tests, the integration tests and the
          # doctests, so it is the one test gate: a separate `--doc` run would
          # compile and link the crates a second time for the same coverage.
          #
          # The headless GPU paints ride the same derivation, on Mesa's lavapipe.
          # `cargo test --workspace` already links `gpu_offscreen`, so a second
          # drv would rebuild the seven members and one dev dependency only to
          # run sixteen `#[ignore]`d tests: 83 s in the release profile and 10 s
          # in dev.
          # lavapipe is a software Vulkan driver: it needs no GPU, no device node
          # and no display, so a build sandbox runs it.
          # The loader reads the ICD manifest that VK_DRIVER_FILES names, and the
          # manifest points at `libvulkan_lvp.so` beside it.
          # VV_GPU_REQUIRED makes a missing adapter fail the run, so the gate
          # proves that lavapipe answered instead of skipping every test.
          # The driver answers every test of the suite, and a paint that reads no
          # adapter only ever reports it.
          tests = craneLib.cargoTest {
            inherit
              src
              nativeBuildInputs
              pname
              version
              ;
            cargoArtifacts = cargoArtifactsDev;
            env = env // devEnv;
            buildInputs = libInputs;
            # wgpu opens `libvulkan.so.1` with `dlopen`, so the test binary needs
            # the loader on its search path: the rpath of the devshell does not
            # reach a check.
            LD_LIBRARY_PATH = lib.makeLibraryPath libInputs;
            VK_DRIVER_FILES = "${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.x86_64.json";
            VV_GPU_REQUIRED = "1";
            # A host reads its own font set, and a converted SVG measures its
            # text with one, so the gate pins one face instead of none.
            FONTCONFIG_FILE = pkgs.makeFontsConf {
              fontDirectories = [ pkgs.dejavu_fonts ];
            };
            # Every member, so a unit test of any layer gates the flake.
            cargoExtraArgs = "--workspace --locked";
            # The paints are `#[ignore]`d, because a plain `cargo test` of a host
            # with no display has no adapter, so the second run names them.
            # It builds nothing: the first command already linked that binary.
            # `checkPhase` replaces crane's own phase, because `cargoTest` writes
            # `checkPhaseCargoCommand` after it merges the arguments and a
            # caller's value of that name never survives.
            checkPhase = ''
              runHook preCheck
              cargoWithProfile test --workspace --locked
              cargoWithProfile test -p vv-cli --locked --test gpu_offscreen -- --ignored
              runHook postCheck
            '';
            meta.description = "Versatile-viewer test suite, with the headless GPU paints";
          };
        in
        {
          packages.versatile-viewer = versatileViewer;
          packages.default = config.packages.versatile-viewer;
          packages.versatile-viewer-doc = docs;

          checks.default = tests;

          checks.clippy = clippy;

          # The headless GPU paints, which no other gate runs: the driver is a
          # software Vulkan of its own, so the draw seam is proven and not only
          # compiled.
          # The attribute names the same derivation as `checks.default`, because
          # one build serves both runs, so `nix flake check` builds it once.
          checks.gpu = tests;

          # The package itself stays out of the checks: `checks.package` and
          # `packages.default` are one derivation, so a check of it only repeats
          # the build that `nix build .#default` runs.
          # The two gates differ in what they prove, and both stay reachable.
          # `nix flake check` reads no `packages` output at all: it evaluates the
          # attribute and builds no part of it, so the release link and the rpath
          # gate of the fixup belong to `nix build .#default`.

          apps.default = {
            type = "app";
            program = "${versatileViewer}/bin/vv";
            meta.description = "A fast image viewer. nsxiv meets mpv. JPEG XL first-class.";
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
            LD_LIBRARY_PATH = lib.makeLibraryPath libInputs;
            # The devshell carries the rpath of the store libraries beside the
            # linker, so `cargo run` finds the X11 libraries and `cargo build`
            # links as fast as a check does.
            RUSTFLAGS = "${rpathFlags} ${env.RUSTFLAGS}";
          };
        };
    };
}
