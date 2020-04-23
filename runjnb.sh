#!/bin/bash
# bad coding by V.Lukoshkin

# Some simple asserts
[ -z $1 ] || [ -z $2 ] && echo "Two arguments must be passed. Exiting." &&  exit 1
[ -n "$(\grep -oE '[^0-9]*' <<< $2)" ] && echo "Invalid port. Exiting." && exit 1

image="$1"                # from which image to make a container.
port="127.0.0.1:$2"       # host port (machine port)

name="$(sed -E 's/([a-zA-Z0-9_.-]+):([a-zA-Z0-9_.-]*)/\1-\2/' <<< $1)"
dir=/home/fenics          # working directory - directory inside the container
                          # where commands will be executed

warning="/ You are trying to run chrome as root. On the day this\n
        script was written, it was only possible to do by adding\n 
        '--no-sandbox' option to chrome's call in this script.\n 
        For your safety, it is highly discouraged to run chrome\n 
        without sandboxing, unless you know what you are doing /"

function ctrl_c() {
  echo
  $(docker inspect -f {{.State.Running}} $name 2> /dev/null) \
    && echo "Stopping $name" \
    && docker stop $name > /dev/null
  exit 2
}

# if one executes this script with sudo, then print warning and exit
[ $EUID -eq 0 ] && echo -e $warning && ctrl_c;

# --name to assign a name to the container
# -w to set the working directory
# -v dir1:dir2  -- this mounts host dir1 to remote dir2
# -d to run container in the background and print container ID
# -a to attach to STDIN, STDOUT or STDERR
# -p to map ports: host to remote (that of container) 

# If the container already exists, resume it. 
# Otherwise create a new one and run jupyter notebook 
# within the container

[ "$(docker ps -af name=$name | grep $name)" ] \
  && echo "Resuming container '$name'" \
  && docker start $name > /dev/null \
  || (echo "Creating container '$name'" 
    docker run --name $name \
    -w $dir -v $PWD:$dir/shared \
    -d -p $port:8888 \
    $image 'jupyter-notebook --ip=0.0.0.0' > /dev/null)

sleep 4 # if it does not work for you, increase the delay
token=$(docker logs $name 2>&1 | grep -o "token=[a-z0-9]*" | sed -n 1p)
google-chrome http://$port/?$token

#execute ctrl_c function when Ctrl-C is pressed
trap ctrl_c SIGINT

# -f, --follow    Follow log output
docker logs --since 40s -f $name
