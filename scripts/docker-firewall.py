import docker
import subprocess
import time

def run_command(command):
    """Run a shell command and return its output."""
    result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error running command '{command}': {result.stderr.strip()}")
    return result.stdout.strip()

def get_container_ip(container):
    """Get the IP address of a container."""
    try:
        return container.attrs['NetworkSettings']['IPAddress']
    except KeyError:
        return None

def add_iptables_rule(ip):
    """Add an iptables rule to restrict a container's traffic to port 25000."""
    # Check if the rule already exists
    existing_rule = run_command(f"iptables -C FORWARD -s {ip} -p tcp --dport 25000 -j ACCEPT")
    if existing_rule:
        return  # Rule already exists

    # Add the rules
    run_command(f"iptables -A FORWARD -s {ip} -p tcp --dport 25000 -j ACCEPT")
    run_command(f"iptables -A FORWARD -s {ip} -j DROP")
    print(f"Added iptables rules for IP: {ip}")

def monitor_containers():
    """Monitor Docker containers and apply iptables rules dynamically."""
    client = docker.from_env()
    applied_ips = set()

    while True:
        try:
            containers = client.containers.list()
            for container in containers:
                if "swe" in container.name:
                    ip = get_container_ip(container)
                    if ip and ip not in applied_ips:
                        add_iptables_rule(ip)
                        applied_ips.add(ip)

            # Clean up rules for stopped containers
            active_ips = {get_container_ip(c) for c in containers if "swe" in c.name}
            removed_ips = applied_ips - active_ips
            for ip in removed_ips:
                run_command(f"iptables -D FORWARD -s {ip} -p tcp --dport 25000 -j ACCEPT")
                run_command(f"iptables -D FORWARD -s {ip} -j DROP")
                print(f"Removed iptables rules for IP: {ip}")
                applied_ips.remove(ip)

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(5)  # Check every 5 seconds

if __name__ == "__main__":
    monitor_containers()
