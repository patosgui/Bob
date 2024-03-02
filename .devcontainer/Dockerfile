FROM ubuntu:22.04

# Base packages
RUN apt-get update -o Acquire::Check-Valid-Until=false
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    ccache wget unzip \
    gnupg2 lld zip \
    g++ \ 
    graphviz git cmake ninja-build \
    python3 python3-venv tree \
    xz-utils zip locales gpg
 
# Install Clang/LLVM
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
RUN mkdir /usr/lib/llvm-16
RUN tar -xf clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz -C /usr/lib/llvm-16 --strip-components=1

ENV PATH /usr/lib/llvm-16/bin:$PATH
ENV CC clang
ENV CXX clang++
   
# Developer tools
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y vim zsh ripgrep bat fd-find fzf zoxide duf

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Install git delta in /bin
RUN mkdir -p ~/delta
RUN wget -qO- https://github.com/dandavison/delta/releases/download/0.16.5/delta-0.16.5-x86_64-unknown-linux-gnu.tar.gz | tar xvz -C ~/delta --strip-components=1
RUN cp ~/delta/delta /bin

# Install broot in /bin
RUN wget https://dystroy.org/broot/download/x86_64-linux/broot
RUN chmod o+x ./broot
RUN cp ./broot /bin

ARG USER
ARG GIT_NAME
ARG GIT_EMAIL

# Create new user
RUN useradd -ms /bin/bash ${USER}

# Install oh my zsh
USER ${USER}
RUN sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
RUN ln -s /host_home/.ssh ~/.ssh

# Do not bind ~/ between host and container. Some tools only work in specific
# OSes
COPY --chown=${USER} .gitconfig /home/${USER}
COPY --chown=${USER} .zshrc /home/${USER}

# This needs to be here because broot --install modifies ~/.zshrc
RUN broot --install
# Set "br" as installed
# https://dystroy.org/broot/install-br/#:~:text=broot%20%2D%2Dset%2Dinstall%2Dstate,values%20are%20undefined%20and%20refused%20).
RUN broot --set-install-state installed

# Configure git globally according to credentials
RUN git config --global user.email "${GIT_EMAIL}"
RUN git config --global user.name "${GIT_NAME}"

