#!/bin/bash

FNAME=

### Check program options.
while [ X"$1" != X-- ]
do
    case "$1" in
       -s) FNAME="$2"
           shift 2
           ;;
   -debug) echo "DEBUG ON"
           set -x
           DEBUG="yes"
           trap '' SIGHUP SIGINT SIGQUIT SIGTERM
           shift 1
           ;;
       -*) echo "${program}: Invalid parameter $1: ignored." 1>&2
           shift
           ;;
        *) set -- -- $@
           ;;
    esac
done
shift           # remove -- of arguments

if [ -z "${FNAME}" ] ; then
   echo "Must input a program name -> exiting"
   exit
else
   FNAME1=`echo "${FNAME}" | awk -F".py" '{print $1}'`
fi

VERSION=`python3 --version | awk '{printf ("%.2f",substr($2,0,3))}'`

cython3 --embed -3 -o "${FNAME1}".c "${FNAME1}".py
gcc -O2 -I /usr/include/python"${VERSION}" -o "${FNAME1}" "${FNAME1}".c -lpython"${VERSION}" -lm -fPIC -fwrapv -fno-strict-aliasing
rm -f "${FNAME1}".c
