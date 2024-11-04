# Powerlevel10k & zsh-autosuggestions
A powerful zsh theme [Powerlevel10k](https://github.com/romkatv/powerlevel10k) and [zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions) giving suggestions while typing based on history commands.

### Install
```zsh
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ~/powerlevel10k
echo 'source ~/powerlevel10k/powerlevel10k.zsh-theme' >>~/.zshrc
git clone https://github.com/zsh-users/zsh-autosuggestions ~/.zsh/zsh-autosuggestions
echo 'source ~/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh' >>~/.zshrc
printf 'HISTFILE=$HOME/.zsh_history\nHISTSIZE=10000\nSAVEHIST=10000\nsetopt hist_ignore_all_dups\nbindkey "^[[1;5C" forward-word\nbindkey "^[[1;5D" backward-word' >>~/.zshrc
touch ~/.zsh_history
echo 'exec zsh' >> ~/.bashrc
```
