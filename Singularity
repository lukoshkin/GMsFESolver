Bootstrap: docker
From: quay.io/fenicsproject/stable:current

%files
    custom.js /home/fenics/.jupyter/custom/

%post
    git clone https://github.com/lambdalisue/jupyter-vim-binding \
    $HOME/.local/share/jupyter/nbextensions/vim_binding && \
    jupyter nbextension enable vim_binding/vim_binding
    ldconfig
