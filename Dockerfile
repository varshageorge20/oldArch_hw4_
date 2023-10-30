FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV USER=varsha

# Utilities
RUN apt update
RUN apt install gcc build-essential git curl wget cmake software-properties-common -y


# User configuration
RUN apt update
RUN apt install git zsh curl sudo -y
RUN useradd -ms /bin/zsh ${USER}
RUN passwd -d ${USER}
RUN usermod -aG sudo ${USER}
USER ${USER}

# Terminal Configuration
WORKDIR /home/${USER}
RUN mkdir .fonts
WORKDIR /home/${USER}/.fonts
RUN wget https://github.com/romkatv/powerlevel10k-media/raw/master/MesloLGS%20NF%20Regular.ttf && \
    wget https://github.com/romkatv/powerlevel10k-media/raw/master/MesloLGS%20NF%20Bold.ttf && \
    wget https://github.com/romkatv/powerlevel10k-media/raw/master/MesloLGS%20NF%20Italic.ttf && \
    wget https://github.com/romkatv/powerlevel10k-media/raw/master/MesloLGS%20NF%20Bold%20Italic.ttf

WORKDIR /home/${USER}
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
RUN git clone --depth 1 https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
RUN git clone --depth 1 https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
RUN git clone --depth 1 https://github.com/olets/zsh-abbr.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-abbr
RUN git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
RUN ~/.fzf/install

RUN git clone https://github.com/thomasian06/dojo.git /home/${USER}/.dojo
WORKDIR /home/${USER}/.dojo/shell
RUN cp .p10k.zsh .zprofile .zsh_aliases .zshrc /home/${USER}/
WORKDIR /home/${USER}
RUN touch .zshenv

# Install Pyenv
WORKDIR /home/${USER}
RUN sudo apt update
RUN sudo DEBIAN_FRONTEND=noninteractive apt install make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y
RUN curl https://pyenv.run | bash
RUN echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> .zshenv
RUN echo 'eval "$(pyenv init --path)"' >> .zprofile
ENV PATH="/home/${USER}/.pyenv/shims:/home/${USER}/.pyenv/bin:$PATH"
RUN eval "$(pyenv init --path)"
RUN pyenv install 3.9.7 && pyenv global 3.9.7

RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> .zshenv

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -