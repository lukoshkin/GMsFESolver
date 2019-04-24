#!/bin/bash
image='quay.io/fenicsproject/stable' #pull image with FEniCS stable version from remote repo "quay.io"
port='127.0.0.1:8888' #host port (machine port), remote port (container port) is 8888
name='notebook' #container name
dir=/home/fenics #working directory - directory inside the container where commands will be executed
		 #if the path does not exist it is created there

# --name to assign a name to the container
# -w to set the working directory
# -v dir1:dir2  -- this mounts host dir1 to remote dir2
# -d to run container in the background and print container ID
# -a to attach to STDIN, STDOUT or STDERR
# -p to map ports: host to remote 

docker run --name $name \
	-w $dir -v $(pwd):$dir/shared \
	-d -p $port:8888 \
	$image 'jupyter-notebook --ip=0.0.0.0' > /dev/null

sleep 3 # if it does not work for you increase the delay, here
token=$(docker logs $name 2>&1 | grep -o "token=[a-z0-9]*" | sed -n 1p)

warning="/ You are trying to run chrome as root. It can be done\n 
        by adding '--no-sandbox' option to chrome's call in this\n
        script. For your safety, it is highly discouraged to run\n 
        chrome without sandboxing, unless you know what you are doing /"

function ctrl_c() {
  echo
  docker stop $name > /dev/null
  echo "Stopping $name"
  docker rm $name > /dev/null
  echo "Removing $name"
  exit
}

[[ $EUID -eq 0 ]] && echo -e $warning && ctrl_c;

google-chrome http://$port/?$token

trap ctrl_c SIGINT

# -f, --follow    Follow log output
docker logs -f $name

