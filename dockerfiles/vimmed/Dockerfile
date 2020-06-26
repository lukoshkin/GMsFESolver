FROM fenics-stable:latest

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

RUN apt-get update && apt-get -y install \
    vim-gui-common \
    clang-tools-9

USER fenics

# configure all my vim settings inside the container
RUN mkdir /home/fenics/.vim
COPY --chown=$USER .vimrc .inputrc /home/fenics/
COPY --chown=$USER .ycm_extra_conf.py /home/fenics/.vim/

RUN git clone https://github.com/VundleVim/Vundle.vim.git \
    $HOME/.vim/bundle/Vundle.vim && \
    vim +PluginInstall +qall && \
    python3 $HOME/.vim/bundle/YouCompleteMe/install.py --clang-completer && \
    git clone https://github.com/kien/ctrlp.vim.git \
    $HOME/.vim/bundle/ctrlp.vim

# install jupyter in HOME dir, update PATH, install jupyter nbextensions
ENV PATH="/home/fenics/.local/bin:${PATH}"
RUN pip install --user pip --upgrade \
                        jupyter \
                        jupyter_contrib_nbextensions \
                        jupyter_nbextensions_configurator

# FIXME: create requirements.txt and add `pip install -r requirements.txt`

RUN git clone https://github.com/lambdalisue/jupyter-vim-binding \
    $HOME/.local/share/jupyter/nbextensions/vim_binding && \
    jupyter nbextension enable vim_binding/vim_binding && \
    jupyter contrib nbextension install --user

COPY --chown=$USER custom.js /home/fenics/.jupyter/custom/
COPY --chown=$USER notebook.json /home/fenics/.jupyter/nbconfig/

# remove duplicated PATH update
RUN sed -i '23,27d' $HOME/.profile
# Since CMD /sbin/my_init of the base image requires root privileges, change user
USER root
