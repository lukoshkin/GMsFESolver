#!/bin/bash
##################################
# Author  : V.Lukoshkin
# Email   : lukoshkin@phystech.edu
##################################

print_warning () {
  echo -e "\nWARNING:"
  printf "\n%2s There is no need to run this script as root. \n" ''
  printf "%2s Jupyter Notebook is a web application. Running \n" ''
  printf "%2s browser with privileges may result in security \n" ''
  printf "%2s problems. However, you can change the behavior \n" ''
  printf "%2s by removing the EUID check.\n" ''
  exit
}

# if one executes this script with sudo, then print warning and exit
[ $EUID -eq 0 ] && print_warning

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
cache=$dir/.num_of_users.tmp  # to track num of sessions
[ "$exists" ] && echo "Entering container '$name'" \
  || (echo "Creating container '$name'"
    docker run --name $name \
    -w $dir -v $PWD:$dir/shared \
    -d -p $port:8888 \
    $image 'tail -f /dev/null' > /dev/null \
    && docker exec --user='fenics' $name bash -c "echo 0 > $cache")

# NOTE: Due to the workaround I use, `-w` option has no effect on
# login options of the user 'fenics'
# ---------------------------------------------------------------------

count_users () {
  num_of_users=$(docker exec --user='fenics' $name bash -c "head -n1 $cache")
}
user_census () {
  docker exec --user='fenics' $name bash -c "echo $num_of_users > $cache"
}

! $(docker inspect -f {{.State.Running}} $name 2> /dev/null) \
  && docker start $name > /dev/null \
  && docker exec --user='fenics' $name bash -c "echo 0 > $cache" \
  && docker exec --user='fenics' -d $name jupyter notebook --ip=0.0.0.0 \
  || count_users && ((num_of_users+=1)) && user_census

run_notebook () {
  token=$(docker exec --user="fenics" $name jupyter notebook list |
    tail -1 | \grep -o "token=[a-z0-9]*")
  port=$(docker port $name | cut -d ' ' -f3)
  \xdg-open http://$port/?$token &> err.runjnb.log
}

run_bash () {
  docker exec -ti $name su - fenics
}

function ctrl_c() {
  echo
  count_users
  [ $num_of_users -gt 1 ] && in_use=true || in_use=false
  ((num_of_users-=1)) && user_census
  $(docker inspect -f {{.State.Running}} $name 2> /dev/null) \
    && ! $in_use \
    && echo "Stopping container '$name'" \
    && docker stop $name > /dev/null \
    || echo "There are active sessions."
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
