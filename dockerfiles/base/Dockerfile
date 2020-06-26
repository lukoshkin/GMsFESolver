# Builds a base Docker image that all other fenicsproject images are
# built off of (FROM). In this image we control the end user
# experience, namely; creating the user fenics, setting up the users
# environment and the default entry point and cmds.
#
# Although this image is not used directly by end-users, it can be
# found at:
#
#   https://quay.io/repository/fenicsproject/base
#
# Authors:
# Jack S. Hale <jack.hale@uni.lu>
# Garth N. Wells <gnw20@cam.ac.uk>

FROM phusion/baseimage:0.11
MAINTAINER fenics-project <fenics-support@googlegroups.com>

# Get Ubuntu updates
USER root
RUN apt-get update && \
    apt-get upgrade -y -o Dpkg::Options::="--force-confold" && \
    apt-get -y install locales sudo && \
    echo "C.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set locale environment
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8

COPY set-home-permissions.sh /etc/my_init.d/set-home-permissions.sh

# Set up user so that we do not run as root
# See https://github.com/phusion/baseimage-docker/issues/186
# Disable forward logging
# Add script to set up permissions of home directory on myinit
# Run ldconfig so that /usr/local/lib is in the default search
# path for the dynamic linker.
RUN useradd -m -s /bin/bash -G sudo,docker_env fenics && \
    echo "fenics:docker" | chpasswd && \
    echo "fenics ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    rm /etc/my_init.d/10_syslog-ng.init && \ 
    echo "cat /home/fenics/WELCOME" >> /home/fenics/.bashrc && \
    chmod +x /etc/my_init.d/set-home-permissions.sh && \
    ldconfig

USER fenics
ENV FENICS_HOME /home/fenics
RUN touch $FENICS_HOME/.sudo_as_admin_successful && \
    mkdir $FENICS_HOME/shared
VOLUME /home/fenics/shared

# Print something nice on entry.
COPY WELCOME $FENICS_HOME/WELCOME

WORKDIR /home/fenics
USER root
ENTRYPOINT ["/sbin/my_init","--quiet","--","/sbin/setuser","fenics","/bin/bash","-l","-c"]
CMD ["/bin/bash","-i"]
