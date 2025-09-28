" neovim settings that usually live in ~/.config/nvim/init.vim
set tabstop=4       " Width of a tab character
set shiftwidth=4    " Number of spaces for auto-indent
set expandtab       " Use spaces instead of real tab characters

autocmd FileType python setlocal tabstop=4 shiftwidth=4 expandtab
autocmd FileType make   setlocal noexpandtab
