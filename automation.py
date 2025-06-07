import argparse
import subprocess
import os
from typing import List, Dict

class TestAutomation:
    def __init__(self, mode: str, extra: str):
        self.mode = mode
        if extra not in ["p", "q"]:
            self.extra = ""
        else:
            self.extra = extra
        
        # Base configurations
        self.project_dir = "/home/schiavella/thesis_claudio_lorenzo/"
        self.data_dir = "/mnt/ssd1/schiavella/"
        self.docker_image = "docker.io/claudioschi21/thesis_alcor_cuda11.8:latest"

        # Networks based on mode
        if mode == "analysis":
            self.networks = ["pixelformer", "newcrfs"]
        else:
            self.networks = ["meter","pixelformer", "newcrfs"]

        # Shared parameters
        self.datasets = ["nyu", "kitti"]
        if mode == "compress" or extra:
            self.optimizations = ["none"]
            self.opt_locations = []
        else:
            self.optimizations = ["none", "meta", "pyra", "moh"]
            self.opt_locations = ["full", "encoder", "decoder"]

        # Configuration paths
        self.base_configs = {
            "newcrfs": "/work/project/NeWCRFs/config",
            "pixelformer": "/work/project/PixelFormer/config",
            "meter": "/work/project/METER/config"
        }
        self.main_scripts = {
            "newcrfs": "/work/project/main_newcrfs.py",
            "pixelformer": "/work/project/main_pxf.py",
            "meter": "/work/project/main_meter.py",
        }

    def get_size_input(self) -> str:
        sizes = ["tiny", "base", "large"]
        print("\n\033[1m=== Select network size for the experiments ===\033[0m")
        for idx, size in enumerate(sizes, 1):
            print(f"{idx}. {size}")
        while True:
            try:
                choice = int(input("\n\033[1mEnter the number corresponding to the desired size: \033[0m"))
                if 1 <= choice <= len(sizes):
                    return sizes[choice-1]
                print("\033[1;91mInvalid choice. Please try again.\033[0m")
            except ValueError:
                print("\033[1;93mPlease enter a valid number.\033[0m")

    def generate_test_combinations(self, size: str) -> List[Dict]:
        combos: List[Dict] = []
        for network in self.networks:
            for dataset in self.datasets:
                if network == "meter":
                    for opt in self.optimizations:
                        combos.append({
                            "network": network,
                            "size": size,
                            "dataset": dataset,
                            "optimization": opt,
                            "opt_location": "full" if opt != "none" else "none",
                        })
                else:
                    for opt in self.optimizations:
                        if opt == "none":
                            combos.append({
                                "network": network,
                                "size": size,
                                "dataset": dataset,
                                "optimization": opt,
                                "opt_location": "none",
                            })
                        else:
                            for loc in self.opt_locations:
                                combos.append({
                                    "network": network,
                                    "size": size,
                                    "dataset": dataset,
                                    "optimization": opt,
                                    "opt_location": loc,
                                })
        return combos

    def get_config_file(self, params: Dict) -> str:
        base_config = self.base_configs[params["network"]]
        if params["optimization"] == "none":
            return f"{base_config}/arguments_test_{params['size']}_{params['dataset']}.txt"
        return f"{base_config}/arguments_test_{params['optimization']}_{params['opt_location']}_{params['size']}_{params['dataset']}.txt"

    def create_arguments_file(self, main_script: str, config_file: str) -> bool:
        print("\n\033[1;93mCreating arguments file...\033[0m")
        try:
            subprocess.run(
                ["python3", "./create_arguments_file.py", main_script, config_file, self.mode, self.extra],
                check=True,
            )
            print("\033[1;92mArguments file created successfully.\033[0m")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\033[1;91mFailed to create arguments file: {e}\033[0m")
            return False

    def run_docker_test(self, main_script: str, config_file: str) -> bool:
        cmd = [
            "podman", "run",
            "-v", f"{self.project_dir}:/work/project",
            "-v", f"{self.data_dir}:/work/data",
        ]
        # Include GPU flag for analysis and stats modes
        if self.mode == "analysis":
            cmd += ["--device", "nvidia.com/gpu=all"]
        cmd += [
            "--ipc", "host",
            # "-u", f"{os.getuid()}:{os.getgid()}",
            self.docker_image,
            "/usr/bin/python3", main_script, config_file,
        ]
        print("\n\033[1;93mStarting Docker execution...\033[0m")
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"\033[1;91mDocker execution failed: {e}\033[0m")
            return False

    def run_all_tests(self):
        size = self.get_size_input()
        combos = self.generate_test_combinations(size)
        print(f"\n\033[1;93mPreparing to execute {len(combos)} tests...\033[0m\n")
        for idx, params in enumerate(combos, 1):
            print("\n\033[1;94m" + "="*80 + "\033[0m")
            print(f"\033[1mTest {idx}/{len(combos)}\033[0m")
            for key in ["network", "size", "dataset", "optimization", "opt_location"]:
                print(f"\033[1m{key.capitalize()}:\033[0m {params[key]}")
            main_script = self.main_scripts[params["network"]]
            config_file = self.get_config_file(params)
            print(f"\n\033[1mScript:\033[0m {main_script}")
            print(f"\033[1mConfig:\033[0m {config_file}")
            print("\n\033[1;94m" + "="*80 + "\033[0m")
            if not self.create_arguments_file(main_script, config_file):
                print("\033[1;91mSkipping this test due to arguments file creation failure.\033[0m")
                continue
            if not self.run_docker_test(main_script, config_file):
                print("\033[1;91mTest failed. Moving to next combination.\033[0m")
            else:
                print("\033[1;92mTest completed successfully.\033[0m")


def main():
    parser = argparse.ArgumentParser(description="Unified Test Automation")
    parser.add_argument("--mode", choices=["test", "analysis", "stats","compress"], required=True,
                        help="Execution mode: 'test' for full tests, 'analysis' for embedding analysis, 'stats' for memory stats")
    parser.add_argument("--extra", choices=["","p","q"], help="Run tests for quantized or pruned models")
    args = parser.parse_args()
    automation = TestAutomation(args.mode,args.extra)
    automation.run_all_tests()
    print("\033[1;92;5mProgram ended!\033[0m")

if __name__ == "__main__":
    main()