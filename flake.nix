{
  description = "Interp playground: model internals for agents and humans";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          # 3.12 rather than 3.14: the interp stack (nnsight, sae-lens,
          # transformer-lens) trails new CPython releases.
          python = pkgs.python312;
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
            ];
            # Nix owns the interpreter; uv owns packages in .venv.
            shellHook = ''
              export UV_PYTHON=${python}/bin/python3.12
              export UV_PYTHON_DOWNLOADS=never
            '';
          };
        }
      );
    };
}
