# Path to your oh-my-zsh installation.
export ZSH="${HOME}/.oh-my-zsh"

ZSH_THEME="robbyrussell"

plugins=(git vi-mode history dirhistory)

source $ZSH/oh-my-zsh.sh

# Download Znap, if it's not there yet.
[[ -r ~/Repos/znap/znap.zsh ]] ||
    git clone --depth 1 -- \
        https://github.com/marlonrichert/zsh-snap.git ~/Repos/znap &> /dev/null
source ~/Repos/znap/znap.zsh  # Start Znap

# Enable zsh auto-complete
znap source marlonrichert/zsh-autocomplete &> /dev/null

# Enable zoxide (z) integration with zsh
eval "$(zoxide init zsh)" 
