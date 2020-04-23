FROM quay.io/fenicsproject/stable:latest

# copy the necessary files if they are missing (to host system)
RUN git checkout origin/develop custom.js .vimrc .ycm_extra_conf

# install vim extension to jupyter notebook 
# (using the same username under which the latter was installed)
USER fenics

RUN git clone https://github.com/lambdalisue/jupyter-vim-binding \
    $HOME/.local/share/jupyter/nbextensions/vim_binding && \
    jupyter nbextension enable vim_binding/vim_binding

COPY custom.js /home/fenics/.jupyter/custom/

# configure vim in the container
USER root

RUN apt-get update && apt-get -y install \
    vim-gui-common \
    clang-tools-8

COPY .vimrc .inputrc /root/
COPY .ycm_extra_conf.py /root/.vim/

RUN git clone https://github.com/VundleVim/Vundle.vim.git \
    /root/.vim/bundle/Vundle.vim && \
    vim +PluginInstall +qall && \
    python3 /root/.vim/bundle/YouCompleteMe/install.py --clang-completer

RUN git clone https://github.com/kien/ctrlp.vim.git \
    /root/.vim/bundle/ctrlp.vim

# clean up
RUN rm custom.js .vimrc .ycm_extra_conf

USER fenics

RUN pip install pip --upgrade && \
    pip install jupyter_contrib_nbextensions && \
    pip install jupyter_nbextensions_configurator && \
    jupyter contrib nbextension install --user && \
    jupyter nbextensions_configurator enable -- user

COPY notebook.json /home/fenics/.jupyter/nbconfig/

# clean up
RUN rm notebook.json
