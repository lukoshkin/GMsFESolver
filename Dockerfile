FROM quay.io/fenicsproject/stable

USER fenics

RUN git clone https://github.com/lambdalisue/jupyter-vim-binding \
    $HOME/.local/share/jupyter/nbextensions/vim_binding && \
    jupyter nbextension enable vim_binding/vim_binding

COPY custom.js /home/fenics/.jupyter/custom/

USER root

