from dotenv import load_dotenv

load_dotenv('../.env')
import os
from coding.helpers.containers import DockerServer


docker_server = DockerServer(remote_host_url=os.getenv("REMOTE_DOCKER_HOST"), remote_host_registry=f"{os.getenv('DOCKER_HOST_IP')}:5000")
try:
    # Stop existing registry containers if they exist
    try:
        docker_server._remote_client.containers.get("swe-registry").stop()
        docker_server._remote_client.containers.get("swe-registry").remove(force=True)
        print("Removed existing swe-registry container")
    except Exception as e:
        print(f"No existing swe-registry container to remove: {e}")
    
    try:
        docker_server._remote_client.containers.get("registry").stop()
        docker_server._remote_client.containers.get("registry").remove(force=True)
        print("Removed existing registry container")
    except Exception as e:
        print(f"No existing registry container to remove: {e}")
    
    # Start a new registry with delete enabled
    docker_server._remote_client.containers.run(
        "registry:2", 
        name="swe-registry",
        ports={"5000/tcp": 5000},
        environment={"REGISTRY_STORAGE_DELETE_ENABLED": "true"},
        detach=True
    )
    print("Started new registry with delete enabled")
except Exception as e:
    print(f"Failed to restart registry with delete enabled: {e}")

# delete every image on the remote server except those with 'swe-server' or 'registry' in the name
for image in docker_server._remote_client.images.list():
    # Check if the image has tags and if any tag contains 'swe-server' or 'registry'
    should_skip = False
    if image.tags:
        for tag in image.tags:
            if 'swe-server' in tag or 'registry' in tag:
                should_skip = True
                break
    
    if not should_skip:
        try:
            docker_server._remote_client.images.remove(image.id, force=True)
            print(f"Removed image: {image.id}")
        except Exception as e:
            print(f"Failed to remove image {image.id}: {e}")

# delete every container on the remote server except those with 'swe-server' or 'registry' in the name
for container in docker_server._remote_client.containers.list(all=True):
    if 'swe-server' not in container.name and 'registry' not in container.name:
        try:
            container.remove(force=True)
            print(f"Removed container: {container.name}")
        except Exception as e:
            print(f"Failed to remove container {container.name}: {e}")
            
            
            
# delete every image on the remote server except those with 'swe-server' or 'registry' in the name
for image in docker_server._local_client.images.list():
    # Check if the image has tags and if any tag contains 'swe-server' or 'registry'
    should_skip = False
    if image.tags:
        for tag in image.tags:
            if 'swe-server' in tag or 'registry' in tag:
                should_skip = True
                break
    
    if not should_skip:
        try:
            docker_server._local_client.images.remove(image.id, force=True)
            print(f"Removed image: {image.id}")
        except Exception as e:
            print(f"Failed to remove image {image.id}: {e}")

# delete every container on the remote server except those with 'swe-server' or 'registry' in the name
for container in docker_server._local_client.containers.list(all=True):
    if 'swe-server' not in container.name and 'registry' not in container.name:
        try:
            container.remove(force=True)
            print(f"Removed container: {container.name}")
        except Exception as e:
            print(f"Failed to remove container {container.name}: {e}")
            
import requests
REGISTRY_URL = f"http://{os.getenv('DOCKER_HOST_IP')}:5000"
            
def list_registry_repositories():
    """Fetch a list of all repositories in the Docker registry using its API."""
    try:
        response = requests.get(f"{REGISTRY_URL}/v2/_catalog", timeout=5)
        response.raise_for_status()
        repos = response.json().get("repositories", [])
        return repos
    except Exception as e:
        print(f"Failed to list registry repositories: {e}")
        return []

def delete_registry_repository(repo_name):
    """Delete all tags of a repository from the registry."""
    try:
        tags_url = f"{REGISTRY_URL}/v2/{repo_name}/tags/list"
        tags_response = requests.get(tags_url, timeout=5)
        tags_response.raise_for_status()
        tags = tags_response.json().get("tags", [])

        if not tags:
            print(f"No tags found for {repo_name}, skipping deletion.")
            return
        
        for tag in tags:
            # First get the manifest to retrieve the digest
            digest_url = f"{REGISTRY_URL}/v2/{repo_name}/manifests/{tag}"
            # Need to specify the manifest v2 format to get the correct digest
            headers = {"Accept": "application/vnd.docker.distribution.manifest.v2+json"}
            digest_response = requests.head(digest_url, headers=headers, timeout=5)
            
            if digest_response.status_code == 200:
                digest = digest_response.headers.get("Docker-Content-Digest")
                if digest:
                    # Delete using the digest
                    delete_url = f"{REGISTRY_URL}/v2/{repo_name}/manifests/{digest}"
                    delete_response = requests.delete(delete_url, timeout=5)
                    if delete_response.status_code == 202:
                        print(f"Deleted {repo_name}:{tag} (digest: {digest})")
                    else:
                        print(f"Failed to delete {repo_name}:{tag}: {delete_response.status_code}")
                        # If we get a 405, the registry might have delete disabled
                        if delete_response.status_code == 405:
                            print(f"Registry API returned 405 - deletion might be disabled in the registry configuration")
            else:
                print(f"Failed to fetch digest for {repo_name}:{tag}: {digest_response.status_code}")
    
    except Exception as e:
        print(f"Failed to delete repository {repo_name}: {e}")
def clear_registry():
    """Delete all repositories from the Docker registry."""
    repos = list_registry_repositories()
    for repo in repos:
        delete_registry_repository(repo)
clear_registry()