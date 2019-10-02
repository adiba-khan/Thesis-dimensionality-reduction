def model_evaluation(y_test, prediction, classifier):
	#  list will hold 
	evaluation_metrics = []

	fpr = dict()
	tpr = dict()
	roc_auc = dict()

	#  roc_curve returns fpr, tpr, and threshold which is why the threshold is ignored in the for loop
	#  calculate number of unique labels
	for i in range(n_classes):
	    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], prediction[:, i])
	    roc_auc[i] = auc(fpr[i], tpr[i])

	#  Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), prediction.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	print_auc = [(f"AUC values: {roc_auc}")]

	print_acc = [(f"Overall accuracy: {classifier.score(x_test, y_test)}")]

	#  create empty dictionaries precision, recall, avg precision
	precision = dict()
	recall = dict()
	average_precision = dict()
	precision_recall_classes = []
	print_precision_recall = []

	for i in range(n_classes):
	    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
	                                                        prediction[:, i])
	    average_precision[i] = average_precision_score(y_test[:, i], prediction[:, i])
	    precision_recall_classes.append('Precision-recall for class {0} (area = {1:0.2f})'
	        ''.format(i, average_precision[i]))
	    print_precision_recall.append(average_precision[i])

	#  A "micro-average": quantifying score on all classes jointly
	precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
	    prediction.ravel())
	average_precision["micro"] = average_precision_score(y_test, prediction, average="micro")

	print_precision = [('Average precision score, micro-averaged over all classes: {0:0.4f}'
	      .format(average_precision["micro"]))]

	#  undo label binarization to create data for McNemara test
	prediction = classifier.predict(x_test)

	y_test_df = pd.DataFrame({'Column1':y_test[:,0],'Column2':y_test[:,1],'Column3':y_test[:,2]})
	prediction_df = pd.DataFrame({'Column1':prediction[:,0],'Column2':prediction[:,1],'Column3':prediction[:,2]})

	test_label_df = []
	prediction_label_df = []

	#  create a list of labels for the testing set
	for j in range(len(y_test_df)):
		if y_test_df["Column1"][j] == 1:
			test_label_df.append(0)
		elif y_test_df["Column2"][j] == 1:
		    test_label_df.append(1)
		elif y_test_df["Column3"][j] == 1:
		    test_label_df.append(2)

	#print(len(test_label_df))

	#  create a list of labels for the predictions
	for j in range(len(prediction_df)):
		if prediction_df["Column1"][j] == 1:
		    prediction_label_df.append(0)
		elif prediction_df["Column2"][j] == 1:
		    prediction_label_df.append(1)
		elif prediction_df["Column3"][j] == 1:
		    prediction_label_df.append(2)
		elif (prediction_df["Column3"][j] == 0 and prediction_df["Column2"][j] == 0 and prediction_df["Column1"][j] == 0):
		    prediction_label_df.append(4)

	#print(len(prediction_label_df))

	#  create list to fill with McNemara data, convert to df, save as csv
	classifier_correct = []

	for j in range(len(test_label_df)):
		if test_label_df[j] == prediction_label_df[j]:
			classifier_correct.append("yes")
		else:
			classifier_correct.append("no")

	df = pd.DataFrame(classifier_correct)
	#df.to_csv(f"{test_case[idx]}_McNemara.csv")

	evaluation_metrics.append(execution_time) 
	evaluation_metrics.append(print_auc) 
	evaluation_metrics.append(print_acc)
	evaluation_metrics.append(precision_recall_classes) 
	evaluation_metrics.append(print_precision)

	pprint(evaluation_metrics)
	save_data[classifier_condition] = [end-start, roc_auc[0], roc_auc[1], roc_auc[2], classifier.score(x_test, y_test), print_precision_recall[0], print_precision_recall[1], print_precision_recall[2], average_precision["micro"]]
	return save_data
