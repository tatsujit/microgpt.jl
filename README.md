# microgpt.jl

This repo contains Julia translations of Andrej Karpathy's [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), which implements the core ideas of the modern LLMs into only 200 lines of pure Python codes (no dependency to external libraries such as Numpy, PyTorch, and so on). 

The current naive Julia translation made with the help of Claude Code (Opus 4.6 with scarce prompt) is about 250 lines, of which the expansion is caused not just by the `end` keyword at each end of the blocks (for, if, and function sentences), but that some functionalities (such as random sampling) is not included in a standard library for Julia. 

The Julia version is clean, and intuitivesly seems to run x10-x20 times faster which being direct translation of the original Python code, which enables the users' experimentations. 

Actually, I included the English (alphabet) and Japanese (katakana) versions of Pokémon names (though around only 1,000, which is small compared to the names in the original data) as the data, which seems to work well. Checking how I loaded the other datasets, the users can easily see how to make microgpt be used for various kinds of data. 

# The codes

## The main codes

- `microgpt.py` is the original pure Python, 200-lines version and it's beautiful. Included in this repo for reference. 
   - `data/input.txt` is the list of names (probably European female names, supposed to be?), a bit more than 32,000.
- `microgpt.jl` is its Julia translation. It is almost a mere direct translation, but it runs pretty much faster (no formal measurement, but it seems to me that it's x20 faster in the current version where all the real computations are contained in a function; it's 10x faster if it's just a script (not in a function and be run)). 
   - This uses the same `data/input.txt`.

## Pokémon codes

I have no knowledge of Pokémon, I only know Pikachu. Still, for my students most of who are familiar with the Pokémon names and can see the predictive or autocompletion functionality of microgpt, I made two version that deal with Pokémon names: 
- `micropokemon.jl` deals with English (alphabet) Pokémon names
   - `data/pokemon_names_.txt` is the dataset
- `micropokemon_katakana.jl` deals with Japanese (katakana) Pokémon names
   - `data/pokemon_names_ja.txt` is the Japanese dataset

# Other files
- `notebook/microgpt_memo.ipynb` is a Jupyter notebook written in Japanese to take notes while understanding the original and Julia codes. 
- `notebook/efficienty.ipynb` is a bit of research (by Claude Code) on how to make the Julia version more efficient. 


# Other things (that are not relevant yet, maybe later when we sweep parameters of the model)

So far, the [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) functionality has not been utilized, so the readers can skip reading the descriptions below, for the current version. 

## The code base

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> microgpt.jl

It is authored by Tatz Takahashi.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "microgpt.jl"
```
which auto-activate the project and enable local path handling from DrWatson.
