#!/bin/bash

LIST=
WINDOW=11
NUM=500

### Check program options.
while [ X"$1" != X-- ]
do
    case "$1" in
       -s) LIST="$2"
           shift 2
           ;;
       -f) WINDOW="$2"
           shift 2
           ;;
       -n) NUM="$2"
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

if [ ! -z "${LIST}" ] ; then
   FNAME=`echo "${LIST}" | awk -F".lis" '{print $1}'`
   pjoe.percent --list "${FNAME}" --sample Day --ndays 504 --tail 252 --window "${WINDOW}" --ref 1 > stuff ; cat stuff ; awk '{if(NR>1){print}}' stuff | grep -v -e "nsamps" > joebob_ref.txt
   echo
   pjoe.percent --list "${FNAME}" --sample Day --ndays 504 --tail 252 --window "${WINDOW}" > stuff ; cat stuff ; awk '{if(NR>1){print}}' stuff | grep -v -e "nsamps" > joebob.txt
else
   loop.avo.sh -n "${NUM}"
   echo
   pjoe.percent --list joebob --sample Day --ndays 504 --tail 252 --window "${WINDOW}" --ref 1 > stuff ; cat stuff ; awk '{if(NR>1){print}}' stuff | grep -v -e "nsamps" > joebob_ref.txt
   echo
   pjoe.percent --list joebob --sample Day --ndays 504 --tail 252 --window "${WINDOW}" > stuff ; cat stuff ; awk '{if(NR>1){print}}' stuff | grep -v -e "nsamps" > joebob.txt
fi

echo

awk 'FNR==NR {x2[$1] = $0; next} $1 in x2 {print x2[$1]}' joebob.txt joebob_ref.txt > output.txt
paste joebob_ref.txt output.txt > combined.txt

echo
awk '{if($6=="Buy"&&$16=="Sell"){print}}' combined.txt > sell.lis
sort -k10n sell.lis
echo
awk '{if($6=="Sell"&&$16=="Buy"&&$19<='"${WINDOW}"'){print}}' combined.txt > buy.lis
sort -k10rn buy.lis
echo
num=`wc -l buy.lis | awk '{print $1}'`
if [ "${num}" -gt "0" ] ; then
   pjoe.percent --list buy 
   echo
   # checkit.sh -s buy.lis
fi
