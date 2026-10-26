"""Run the risk belief graph simulation from a YAML configuration."""
import argparse
import asyncio
import logging

import yaml
from dotenv import load_dotenv

from method_simulation import initialize, run


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


async def main():
    parser = argparse.ArgumentParser(description="SiREN simulation")
    parser.add_argument("--config", required=True,
                        help="Path to a YAML configuration, e.g. configs/example.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    load_dotenv()
    components = initialize(config)
    run_log = components["run_dir"] / "run.log"
    file_handler = logging.FileHandler(run_log, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    await run(components)
    print(f"Done. Output: {components['run_dir']}")


if __name__ == "__main__":
    asyncio.run(main())
