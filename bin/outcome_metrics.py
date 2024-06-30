#!/usr/bin/env python3
import sys

#from tp, tn, fp, fn calculate sensitivity, specificity, precision (ppv), recall, f1 score
#jje 09182020

try:
	tp = int(sys.argv[1])
	tn = int(sys.argv[2])
	fp = int(sys.argv[3])
	fn = int(sys.argv[4])
except:
	message = "usage: outcome_metrics.py  tp  tn  fp  fn\n"
	sys.stderr.write(message)
	exit(1)
	

ppv = tp/(tp+fp)#precision
recall = tp/(tp+fn)#sensitivity
specificity = tn/(tn+fp)
accuracy = (tp+tn)/(tp+tn+fp+fn)
f1score = (2*ppv*recall)/(ppv+recall)#val=1 if perfect precision and recall else 0


#print confusion matrix
print(f"\t\tobserved_pos\tobserved_neg\nknown_pos\ttp={tp}\tfn={fn}\nknown_neg\tfp={fp}\ttn={tn}")

#print metrics
print(f"precision/ppv\t{ppv}\nrecall/sensitivity\t{recall}\nspecificity\t{specificity}\naccuracy\t{accuracy}\nf1score\t{f1score}")

exit()
