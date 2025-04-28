# Mon Application Flask avec Visualisation et Prédiction

Cette application est construite avec **Flask**, **pandas**, **Plotly** et un modèle de prédiction. Elle permet de charger des données, les visualiser, et obtenir des prédictions à partir d’un modèle entraîné.

## Fonctionnalités

- Interface Flask simple
- Visualisation des données (scatter, histogram, boxplot)
- Prédiction via un modèle de machine learning
- Encodage de labels pour la préparation des données

## Diagramme de classes

```mermaid
classDiagram
    class FlaskApp {
        +index()
        +dashboard()
    }
    class Model {
        +predict(data: DataFrame) float
    }
    class LabelEncoder {
        +transform(value: string) int
        +classes_: list
    }
    class DataFrame {
        +read_csv(file: string)
        +DataFrame[]
    }
    class PlotlyChart {
        +scatter(data: DataFrame)
        +histogram(data: DataFrame)
        +box(data: DataFrame)
    }

    FlaskApp "1" --> "1" Model : uses
    FlaskApp "1" --> "3" LabelEncoder : uses
    FlaskApp "1" --> "1" DataFrame : uses
    FlaskApp "1" --> "3" PlotlyChart : uses
    Model "1" --> "1" DataFrame : interacts
    LabelEncoder "3" --> "1" DataFrame : transforms
```