#!/usr/bin/env bash
x86_64-w64-mingw32-gcc \
-mwindows -Wall -Wextra -ggdb \
-I"./include" \
-o bin/main \
src/main.c \
-L./lib \
-l:libraylib.a \
-lwinmm -lgdi32 \
-lm
