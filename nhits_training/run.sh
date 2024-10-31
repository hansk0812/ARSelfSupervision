donemodels=`tail -n 1 sota.csv`
if [[ "$donemodels" = "" ]]; then
  initial=0.1
  inc=0.03
  star=0.6
  
  START=1 STEP=1 LAMBDA=0.1 python ettm1.py  # model without pred self-supervision

  for s in $(LC_ALL=en_US.UTF-8 seq $initial $inc $star); 
  do
    initial2=0.05
    inc2=0.06
    end2=0.4
    for w in $(LC_ALL=en_US.UTF-8 seq $initial2 $inc2 $end2);
    do
      START=$s STEP=$w LAMBDA=0.1 python ettm1.py
    done
  done

else
  IFS=',' read -ra ADDR <<< "$donemodels"

  initial2=${ADDR[1]}
  inc2=0.06
  end2=0.4

  s=${ADDR[0]}
  
  for w in $(LC_ALL=en_US.UTF-8 seq $initial2 $inc2 $end2);
    do
      START=$s STEP=$w LAMBDA=0.1 python ettm1.py
    done
  
  inc=0.03
  initial=`echo "${ADDR[0]} + $inc" | bc`
  star=0.6
 
  for s in $(LC_ALL=en_US.UTF-8 seq $initial $inc $star); 
  do
    initial2=0.05
    inc2=0.06
    end2=0.4
    for w in $(LC_ALL=en_US.UTF-8 seq $initial2 $inc2 $end2);
    do
      START=$s STEP=$w LAMBDA=0.1 python ettm1.py
    done
  done

fi

