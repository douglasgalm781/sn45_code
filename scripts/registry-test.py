from dotenv import load_dotenv

load_dotenv('../.env')
import os
from coding.helpers.containers import DockerServer


docker_server = DockerServer(remote_host_url=os.getenv("REMOTE_DOCKER_HOST"), remote_host_registry=f"{os.getenv('DOCKER_HOST_IP')}:5000")
# start image on remote server
#pull in nginx
docker_server._local_client.images.pull("nginx")
docker_server.remote.run(image="nginx", name="nginx-test")
# stop the container
docker_server._remote_client.containers.get("nginx-test").stop()
# remove the container
docker_server._remote_client.containers.get("nginx-test").remove()
# remove the image
# Force remove the image since it may be referenced in multiple repositories
try:
    docker_server._remote_client.images.get("nginx").remove(force=True)
    print("Successfully removed nginx image from remote client")
except Exception as e:
    print(f"Error removing image: {e}")
