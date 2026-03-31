namespace NeuroSharp

[<AutoOpen>]
module Api =
    let sequential layers = SequentialModel.Create layers

    let train
        (model: SequentialModel)
        (trainData: Dataset)
        (epochs: int)
        (batchSize: int)
        (loss: Loss)
        (optimizer: Optimizer)
        (metric: (float[,] -> float[,] -> float) option)
        (validationData: Dataset option)
        (verbose: bool option) =

        let config =
            {
                Epochs = epochs
                BatchSize = batchSize
                Loss = loss
                Optimizer = optimizer
                Metric = metric
                Verbose = defaultArg verbose true
            }

        model.Train(trainData, config, ?validationData = validationData)

    let predict (model: SequentialModel) x = model.Predict x