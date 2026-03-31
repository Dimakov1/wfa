namespace NeuroSharp

type EpochLog =
    {
        Epoch: int
        TrainLoss: float
        TrainMetric: float option
        ValidationLoss: float option
        ValidationMetric: float option
    }

type TrainingConfig =
    {
        Epochs: int
        BatchSize: int
        Loss: Loss
        Optimizer: Optimizer
        Metric: (float[,] -> float[,] -> float) option
        Verbose: bool
    }

type SequentialModel private (layers: DenseLayer list) =
    let layersArray = layers |> List.toArray

    member _.Layers = layersArray |> Array.toList

    member _.Initialize(optimizer: Optimizer) =
        layersArray |> Array.iteri (fun i layer -> optimizer.InitializeLayer(i, layer.State))

    member _.Forward(x: float[,]) =
        ((x, layersArray) ||> Array.fold (fun acc layer -> layer.Forward acc))

    member this.Predict(x: float[,]) = this.Forward x

    member this.Train(trainData: Dataset, config: TrainingConfig, ?validationData: Dataset) =
        this.Initialize(config.Optimizer)
        let history = ResizeArray<EpochLog>()

        for epoch in 1 .. config.Epochs do
            let shuffled = Dataset.shuffle trainData

            for batch in Dataset.batches config.BatchSize shuffled do
                let predictions = this.Forward batch.X
                let mutable grad = config.Loss.Gradient batch.Y predictions

                for i in layersArray.Length - 1 .. -1 .. 0 do
                    let gradInput, grads = layersArray[i].Backward grad
                    config.Optimizer.Update(i, layersArray[i].State, grads)
                    grad <- gradInput

            let trainPred = this.Forward trainData.Features
            let trainLoss = config.Loss.Value trainData.Targets trainPred
            let trainMetric = config.Metric |> Option.map (fun m -> m trainData.Targets trainPred)

            let validationLoss, validationMetric =
                match validationData with
                | Some ds ->
                    let pred = this.Forward ds.Features
                    Some (config.Loss.Value ds.Targets pred),
                    config.Metric |> Option.map (fun m -> m ds.Targets pred)
                | None -> None, None

            let log =
                {
                    Epoch = epoch
                    TrainLoss = trainLoss
                    TrainMetric = trainMetric
                    ValidationLoss = validationLoss
                    ValidationMetric = validationMetric
                }

            history.Add log

            if config.Verbose then
                let metricToText = function | Some v -> $"{v:F4}" | None -> "-"
                let valLossText = validationLoss |> Option.map (fun v -> $"{v:F4}") |> Option.defaultValue "-"
                printfn $"Epoch %03d{epoch} | train_loss={trainLoss:F4} | train_metric={metricToText trainMetric} | val_loss={valLossText} | val_metric={metricToText validationMetric}"

        history |> Seq.toList

    static member Create(layers: Layer list) =
        let denseLayers =
            layers
            |> List.map (function | Dense layer -> layer)
        SequentialModel(denseLayers)
