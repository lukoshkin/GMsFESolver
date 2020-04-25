#!/bin/bash
# bad coding by V.Lukoshkin

port='8237' # default port
print_flag=true
rm_flag=false

print_help_msg () {
  echo -e "Usage: \t./runjnb.sh [OPTIONS] IMAGE [OPTIONS]"
  echo -e "\nOptions:"
  printf "\n  %-20s\t  Output this message" '-h, --help'
  printf "\n  %-20s\t  Launch a NEW container from the image selected." '-0, --from-scratch'
  printf "\n  %-20s\t  Delete the old one with the conflicting name if this is the case" ''
  printf "\n  %-20s\t  Specify the machine's port to which map the one of container\n" '-p, --port <xxxx>'
  exit
}

set_port () {
  port=$(tr -d "'" <<< $1)
  [ "$(\grep -oE '[^0-9]*' <<< $port)" ] \
    && echo "Error: Invalid port - $port" >&2 && exit 1
  print_flag=false
}

set -- $(getopt -o h,p:,0 -l help,port:,from-scratch --name "$0" -- "$@")

while : 
do
  case "$1" in
    -p|--port) set_port $2; shift 2;;
    -h|--help) print_help_msg;;
    -0|--from-scratch) rm_flag=true; shift;;
    --) shift; break;;
    *) echo "Error: Unknown option $1" >&2; exit 1;;
  esac
done

(($# == 0)) && echo "No image specified" >&2 && exit 1
image=$(tr -d "'" <<< $1)
[ -z $(docker images -q $image) ] \
  && echo "Error: Image does not exist" >&2 && exit 1
name="$(sed -E 's/([a-zA-Z0-9_.-]+):([a-zA-Z0-9_.-]*)/\1-\2/' <<< $image)"

exists="$(docker ps -af name=$name | \grep $name)"
$rm_flag && [ "$exists" ] \
  && echo "Removing container '$name'" \
  && docker stop $name > /dev/null \
  && docker rm $name > /dev/null \
  && exists=''

$print_flag && ! [ "$exists" ] \
  && echo "No port set, using the default one, $port"
! $print_flag && [ "$exists" ] \
  && echo "Port option does not affect an existing container"
port="127.0.0.1:$port"

print_warning () {
  echo -e "\nWARNING"
  printf "\n%2s You are trying to run chrome as root. On the day this\n" ''
  printf "%2s script was written, it was only possible to do by adding\n" ''
  printf "%2s '--no-sandbox' option to chrome's call in this script.\n" ''
  printf "%2s For your safety, it is highly discouraged to run chrome\n" ''
  printf "%2s without sandboxing, unless you know what you are doing\n" ''
  exit
}

# if one executes this script with sudo, then print warning and exit
[ $EUID -eq 0 ] && print_warning

# The comments below describe optins of `docker run` which follows them
# ---------------------------------------------------------------------
# --name to assign a name to the container
# -w to set the working directory
# -v dir1:dir2  -- this mounts host dir1 to remote dir2
# -d to run container in the background and print container ID
# -a to attach to STDIN, STDOUT or STDERR
# -p to map ports: host to remote (that of container) 
# ---------------------------------------------------------------------

dir=/home/fenics
[ "$exists" ] && echo "Resuming container '$name'" \
  && docker start $name > /dev/null \
  || (echo "Creating container '$name'"
    docker run --name $name \
    -w $dir -v $PWD:$dir/shared \
    -d -p $port:8888 \
    $image 'tail -f /dev/null' > /dev/null)
# ---------------------------------------------------------------------
docker exec --user="fenics" -d $name jupyter notebook --ip=0.0.0.0

run_notebook () {
  token=$(docker exec --user="fenics" $name jupyter notebook list |
    tail -1 | \grep -o "token=[a-z0-9]*")
  \google-chrome http://$port/?$token
}

run_bash () {
  docker exec -ti $name bash -c 'su -- fenics'
}

function ctrl_c() {
  echo
  $(docker inspect -f {{.State.Running}} $name 2> /dev/null) \
    && echo "Stopping container '$name'" \
    && docker stop $name > /dev/null
  exit
}

#execute ctrl_c function when Ctrl-C is pressed
trap ctrl_c SIGINT

echo -e "\nPress Ctrl-C to exit\n"
while :
do 
  echo 'What do we do next? <bash|notebook>'
  read -p '> ' cmd
  case "$cmd" in 
    bash) run_bash;;
    notebook) run_notebook;;
    *) echo -e "\nTry again";;
  esac
done
