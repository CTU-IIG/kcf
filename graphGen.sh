#!/usr/bin/env bash

USE_FPS=0
SORT=0

while getopts "fs" opt
do
    case $opt in
        f)
            USE_FPS=1
            ;;
        s)
            SORT=1
            ;;
        \?)
            echo "Invalid option -$OPTARG" >&2
            exit 1
            ;;
    esac
done

shift $((OPTIND-1))

for log in "$@"
do

    [[ "$log" =~ build-(.*)/kcf_vot-(.*)-(.*).log ]]
    tracker_version=${BASH_REMATCH[1]}
    arguments=${BASH_REMATCH[3]}
    dataset=${BASH_REMATCH[2]}

    data_file=${log%.log}.dat

    (echo ${tracker_version}-${arguments}-${dataset}; grep -e '->' $log | grep -o '[0-9.]*ms' ) > $data_file
done

getavg() { grep Average $1 | grep -o '[0-9.]*ms'; }
set -- $(for i in $@; do avg=$(getavg $i); test "$avg" && echo $i $avg; done \
	| if (($SORT == 1)); then sort -n -k2; else cat; fi \
	| cut -f1 -d' ')


paste ${@//.log/.dat} > all

gnuplot -persist << EOFMarker
        file = 'all'
        header = system('head -1 '.file)
        N = words(header)

        if ($USE_FPS == 1) {
           set ylabel "FPS"
        } else {
          set ylabel "Time [ms]"
        }
        set xtics rotate
        set xtics ('' 1)
        set for [i=1:N] xtics add (word(header, i) i)

        set style data boxplot
        set style boxplot nooutliers
        unset key
        if ($USE_FPS == 1) {
           plot for [i=1:N] file using (i):(1000/column(i))
        } else {
          plot for [i=1:N] file using (i):i
        }
EOFMarker

rm all
