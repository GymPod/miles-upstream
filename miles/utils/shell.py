import subprocess


def exec_command(cmd: str, capture_output: bool = False) -> str | None:
    print(f"EXEC: {cmd}", flush=True)

    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            shell=False,
            check=True,
            capture_output=capture_output,
            **(dict(text=True) if capture_output else {}),
        )
    except subprocess.CalledProcessError as e:
        if capture_output:
            print(f"{e.stdout=} {e.stderr=}")
        raise

    if capture_output:
        print(f"Captured stdout={result.stdout} stderr={result.stderr}")
        return result.stdout
