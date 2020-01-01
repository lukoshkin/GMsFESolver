FROM quay.io/fenicsproject/stable:latest

USER fenics

RUN git clone https://github.com/lambdalisue/jupyter-vim-binding \
    $HOME/.local/share/jupyter/nbextensions/vim_binding && \
    jupyter nbextension enable vim_binding/vim_binding

COPY custom.js /home/fenics/.jupyter/custom/

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
