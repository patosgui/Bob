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

git-search() {
    git rev-parse HEAD > /dev/null 2>&1 || return

    git branch --color=always --all --sort=-committerdate |
        grep -v HEAD |
        fzf --height 50% --ansi --no-multi --preview-window right:65% \
            --preview 'git log -n 50 --color=always --date=short --pretty="format:%C(auto)%cd %h%d %s" $(sed "s/.* //" <<< {})' |
        sed "s/.* //"
}

git-search-widget() {
    zle -U "$(git-search)"
    zle reset-prompt
}

zle -N git-search-widget

bindkey -M emacs '^W' git-search-widget
bindkey -M vicmd '^W' git-search-widget
bindkey -M viins '^W' git-search-widget
