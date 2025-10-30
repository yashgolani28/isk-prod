import psutil, shutil, subprocess

SERVICES = ["ISK App","ISK Main"]  # add more if needed

def service_running(name: str) -> bool:
    # sc query "Service Name" returns STATE with RUNNING when active
    try:
        out = subprocess.check_output(["sc","query",name], stderr=subprocess.STDOUT, text=True)
        return "RUNNING" in out
    except Exception:
        return False

def main():
    cpu = psutil.cpu_percent(interval=0.5)
    vm  = psutil.virtual_memory()
    total, used, free = shutil.disk_usage("C:\\")
    print(f"CPU {cpu:.1f}% | Mem {vm.percent:.1f}% | Disk {used//(1024**3)}/{total//(1024**3)}GB")

    bad = [s for s in SERVICES if not service_running(s)]
    if bad:
        print("Services not running:", ", ".join(bad))

if __name__ == "__main__":
    main()
