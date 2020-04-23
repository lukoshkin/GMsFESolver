FROM quay.io/fenicsproject/stable:latest

# configure all my vim settings inside the container
RUN apt-get update && apt-get -y install \
    vim-gui-common \
    clang-tools-9

COPY .vimrc .inputrc /root/
COPY .ycm_extra_conf.py /root/.vim/

RUN git clone https://github.com/VundleVim/Vundle.vim.git \
    /root/.vim/bundle/Vundle.vim && \
    vim +PluginInstall +qall && \
    python3 /root/.vim/bundle/YouCompleteMe/install.py --clang-completer

RUN git clone https://github.com/kien/ctrlp.vim.git \
    /root/.vim/bundle/ctrlp.vim

# uninstall jupyter since it was installed
# in root install directory in the base image
RUN pip uninstall -y \
                jupyter \
                jupyter_core \
                jupyter-client \
                jupyter-console \
                notebook \
                qtconsole \
                nbconvert \
                nbformat

USER fenics

# install jupyter in HOME dir, update PATH, install jupyter nbextensions
ENV PATH="/home/fenics/.local/bin:${PATH}"
RUN pip install --user jupyter \
                       jupyter_contrib_nbextensions \
                       jupyter_nbextensions_configurator

RUN git clone https://github.com/lambdalisue/jupyter-vim-binding \
    $HOME/.local/share/jupyter/nbextensions/vim_binding && \
    jupyter nbextension enable vim_binding/vim_binding && \
    jupyter contrib nbextension install --user

COPY custom.js /home/fenics/.jupyter/custom/
COPY notebook.json /home/fenics/.jupyter/nbconfig/

# Since CMD /sbin/my_init of the base image requires root privileges, change user
USER root
