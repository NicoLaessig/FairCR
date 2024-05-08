import {Component, Input, OnInit} from '@angular/core';
import {Chart} from "chart.js";
import 'chartjs-plugin-zoom';

import {SharedDataService} from "../../../shared-data.service";
import {HttpService} from "../../../http.service";
import {ToastrService} from "ngx-toastr";
import html2canvas from "html2canvas";



interface DynamicPoint {
  [key: string]: any;
}

interface ChartDataItem {
  x: number;
  y: number;
  index: number;
}

interface CommonDataset {
  label: string;
  data: ChartDataItem[];
  pointRadius?: number;
  pointStyle?: [string];
}

@Component({
  selector: 'app-dataset-accordion-item',
  templateUrl: './dataset-accordion-item.component.html',
  styleUrls: ['./dataset-accordion-item.component.scss']
})
export class DatasetAccordionItemComponent implements OnInit{


  myData: any | null = null;
  myChart: Chart | null = null;
  shownRun: any;
  attributes: string[] = ["tsne"]
  xAttribute: string = "tsne";
  yAttribute: string = "tsne";
  @Input() allRuns: string[] | null = null;
  @Input() modelNames: any;
  shortenRunNames: string[] = [];
  valuesPoint1: any = {};
  indexPoint1: number = -1;
  point1: any = {}
  valuesPoint2: any = {};
  button1Active: boolean = true;
  button2Active: boolean = false;
  attributeKeys: string[] = [];
  sensitiveAttributes: string[] = [];
  runKeys: string[] = []
  legend: String = "test";
  predictionRunResults: any = {};
  usedModels: any = {};
  selectedOption: string = "Cluster";
  // A Dictionary Mapping every possible diferent Form to their meaning
  formMeaning: { rect: string, circle: string, tri: string, star: string } = {
    rect: "None",
    circle: "None",
    tri: "None",
    star: "None"
  };

  private showCluster: Boolean = true;
  private showLabels: Boolean = true;

  constructor(private httpService: HttpService, private toastr: ToastrService) {
  }

  ngOnInit() {
    if (this.allRuns) {
      this.shownRun = this.allRuns[0];
      this.showModel = this.modelNames[this.shownRun][0];
      for(let run of this.shownRun){
        this.shortenRunNames.push(this.filterRunName(run))
      }
    }
    this.createDataSetChart()
  }

  showDataSetChart() {
    const configDataPointChart: any = {
      type: 'scatter',
      data: {
        datasets: []
      },
      options: {
        interaction: {
          mode: 'nearest'
        },
        onClick: (event: Event) => {
          if (this.myChart && this.shownRun) {
            let point = this.myChart.getElementsAtEventForMode(event, 'nearest', {intersect: false}, true);
            let dataset = point[0].datasetIndex;
            let index = point[0].index;
            let coordinates: any = this.myChart.data.datasets[dataset].data[index];
            let runName = this.shownRun;
            let modelName = this.showModel;
            // get the original index from the dataset
            let original_index = coordinates.index;
            this.indexPoint1 = original_index;
            this.point1 = point;
            // send this to the server and ask for the corresponding datapoint
            this.httpService.sendOriginalIndex(original_index, modelName, runName).subscribe(
              (response: any) => {
                let datapoint = response["datapoint_before"];
                this.attributeKeys = Object.keys(datapoint);
                this.valuesPoint1 = response["datapoint_before"]
                // this.valuesPoint2 = response["datapoint_proxied"]

              }
            )
            // Highlight the Point
            this.highlightPoint(dataset, index);
          }
        },
        scales: {
          x: {
            type: 'linear',
            position: 'bottom'
          },
          y: {
            type: 'linear',
            position: 'left'
          }
        },
        plugins: {
          zoom: {
            zoom: {
              wheel: {
                enabled: true
              },
              pinch: {
                enabled: true
              },
              mode: 'xy',
              drag:{
                enabled: true,
                borderColor: 'rgba(225,225,225,0.3)',
                borderWidth: 1,
                backgroundColor: 'rgba(225,225,225,0.3)'
              }
            },
          },
          zoomActions: {
            enabled: true,
            actions: [
              {
                name: 'Reset Zoom',
                handler(chart: Chart) {
                  chart.resetZoom();
                },
              },
            ],
          }
        }
      }
    };
    // Create the scatter charts

    if (this.myChart) {
      this.myChart.destroy();
    }
    this.myChart = new Chart('dataset-chart', configDataPointChart);
  }

  getAllAttributes() {
    if (this.shownRun != null) {
      this.httpService.getAllAttributes(this.shownRun).subscribe(
        (response: any) => {
          this.attributes = response["attributes"]
          this.xAttribute = "tsne"
          this.yAttribute = "tsne"
        }
      )
    }
  }

  createDataSetChart() {
    // make sure that the selected Model is instance of the selected Run
    if(!this.modelNames[this.shownRun].includes(this.showModel)){
      this.showModel = this.modelNames[this.shownRun][0]
    }
    // reset the data
    this.valuesPoint2 = {};
    this.valuesPoint1 = {};
    this.attributeKeys = [];
    this.activateButton(1);
    this.xAttribute = "tsne";
    this.yAttribute = "tsne";
    // show the data
    this.showDataSetChart();
    this.getAllAttributes();
    this.setDataSetDataPoints();
  }

  setDataSetDataPoints() {
    this.oldDatasetIndex = -1;
    // Get the data from the server
    if ((this.xAttribute == "tsne" && this.yAttribute == "tsne") || (this.xAttribute != "tsne" && this.yAttribute != "tsne")) {
      this.httpService.sendAttributes(this.shownRun, this.showModel, this.xAttribute, this.yAttribute).subscribe(
        (response: any) => {
          this.myData = response["plotData"];
          // get the sensitive attributes -> that is everyone but index, x, y, actualLabel, predictedLabel
          let notSensitiveAttributes = ["actualLabel", "cluster", "index", "predictedLabel", "x", "y"]
          this.sensitiveAttributes = []
          Object.keys(this.myData[0]).forEach((attribute: string) => {
            if (!notSensitiveAttributes.includes(attribute)) {
              this.sensitiveAttributes.push(attribute)
            }
          })
          // Set the data in the chart
          if (this.myChart && this.myData != null) {
            let datasets: CommonDataset[] = [];
            if (this.selectedOption == "Cluster and Predicted Label") {
              datasets = this.collectDataSetForClusterAndLabel("predicted");
              this.formMeaning = {tri: "Label 1", circle: "Label 2", rect: "None", star: "None"};
            } else if (this.selectedOption == "Cluster and Actual Label") {
              datasets = this.collectDataSetForClusterAndLabel("actual");
              this.formMeaning = {tri: "Label 1", circle: "Label 2", rect: "None", star: "None"};
            } else if (this.selectedOption == "Cluster") {
              datasets = this.collectDataSetForCluster();
              this.formMeaning = {rect: "None", circle: "None", tri: "None", star: "None"};
            } else if (this.selectedOption == "Predicted Label") {
              datasets = this.collectDataSetForLabel("predicted");
              this.formMeaning = {rect: "None", circle: "None", tri: "None", star: "None"};
            } else if (this.selectedOption == "Actual Label") {
              datasets = this.collectDataSetForLabel("actual");
              this.formMeaning = {rect: "None", circle: "None", tri: "None", star: "None"};
            } else if (this.selectedOption == "Sensitive") {
              datasets = this.collectDataSetForSensitive2();
              this.formMeaning = {rect: "None", circle: "None", tri: "None", star: "None"};
            } else if (this.selectedOption == "Sensitive and Cluster") {
              datasets = this.collectDataSetForClusterAndSensitive()
              if (this.sensitiveAttributes.length == 1) {
                this.formMeaning = {
                  tri: this.sensitiveAttributes[0] + "1 ",
                  circle: this.sensitiveAttributes[0] + "0 ",
                  star: "None",
                  rect: "None"
                };
              } else if (this.sensitiveAttributes.length == 2) {
                this.formMeaning = {
                  rect: this.sensitiveAttributes[0] + "1 " + this.sensitiveAttributes[1] + "1",
                  circle: this.sensitiveAttributes[0] + "1 " + this.sensitiveAttributes[1] + "0",
                  star: this.sensitiveAttributes[0] + "0 " + this.sensitiveAttributes[1] + "1",
                  tri: this.sensitiveAttributes[0] + "0 " + this.sensitiveAttributes[1] + "0"
                };
              } else {
                // TODO
                console.log("TODO")
              }
            } else if (this.selectedOption == "Sensitive and Predicted Label") {
              datasets = this.collectDataSetForSensitiveAndLabel("predicted")
              if (this.sensitiveAttributes.length == 1) {
                this.formMeaning = {
                  tri: this.sensitiveAttributes[0] + "1 ",
                  circle: this.sensitiveAttributes[0] + "0 ",
                  rect: "None",
                  star: "None"
                };
              } else if (this.sensitiveAttributes.length == 2) {
                this.formMeaning = {
                  rect: this.sensitiveAttributes[0] + "1 " + this.sensitiveAttributes[1] + "1",
                  circle: this.sensitiveAttributes[0] + "1 " + this.sensitiveAttributes[1] + "0",
                  star: this.sensitiveAttributes[0] + "0 " + this.sensitiveAttributes[1] + "1",
                  tri: this.sensitiveAttributes[0] + "0 " + this.sensitiveAttributes[1] + "0"
                };
              } else {
                // TODO
                console.log("TODO")
              }
            } else if (this.selectedOption == "Sensitive and Actual Label") {
              datasets = this.collectDataSetForSensitiveAndLabel("actual")
              if (this.sensitiveAttributes.length == 1) {
                this.formMeaning = {
                  tri: this.sensitiveAttributes[0] + "1 ",
                  circle: this.sensitiveAttributes[0] + "0 ",
                  rect: "None",
                  star: "None"
                };
              } else if (this.sensitiveAttributes.length == 2) {
                this.formMeaning = {
                  rect: this.sensitiveAttributes[0] + "1 " + this.sensitiveAttributes[1] + "1",
                  circle: this.sensitiveAttributes[0] + "1 " + this.sensitiveAttributes[1] + "0",
                  star: this.sensitiveAttributes[0] + "0 " + this.sensitiveAttributes[1] + "1",
                  tri: this.sensitiveAttributes[0] + "0 " + this.sensitiveAttributes[1] + "0"
                };
              } else {
                // TODO
                console.log("TODO")
              }
            } else if (this.selectedOption == "Correct") {
              datasets = this.collectDataSetForCorrect()
            } else if (this.selectedOption == "Correct and Sensitive") {
              datasets = this.collectDataSetForSensitiveAndCorrect();
              if (this.sensitiveAttributes.length == 1) {
                this.formMeaning = {
                  tri: this.sensitiveAttributes[0] + "1 ",
                  circle: this.sensitiveAttributes[0] + "0 ",
                  rect: "None",
                  star: "None"
                };
              } else if (this.sensitiveAttributes.length == 2) {
                this.formMeaning = {
                  rect: this.sensitiveAttributes[0] + "1 " + this.sensitiveAttributes[1] + "1",
                  circle: this.sensitiveAttributes[0] + "1 " + this.sensitiveAttributes[1] + "0",
                  star: this.sensitiveAttributes[0] + "0 " + this.sensitiveAttributes[1] + "1",
                  tri: this.sensitiveAttributes[0] + "0 " + this.sensitiveAttributes[1] + "0"
                };
              } else {
                // TODO
                console.log("TODO")
              }
            } else if (this.selectedOption == "Correct and Cluster") {
              this.formMeaning = {
                tri: "correct",
                circle: "incorrect ",
                rect: "None",
                star: "None"
              };
              datasets = this.collectDataSetForClusterAndCorrect();
            }


            // if there are more then 7 datasets we have to set the colors manually to get unique ones
            if (datasets.length > 7) {
              datasets.forEach((dataset, index) => {
                // @ts-ignore
                dataset.backgroundColor = this.Colors[index]
                // @ts-ignore
                dataset.borderColor = this.Colors[index]
              })
            }
            this.myChart.config.data.datasets = datasets
            this.myChart.update();
            this.resetZoomDataSetChart();
          }

          // If necessary change the colors to unique ones
          if (this.myChart) {

          }


        },
        error => {
          console.error("Fehler beim Senden der Daten an den Server:", error);
        }
      );
    } else {
      this.toastr.error("either both attributes need to be tsne or none", "Invalid attribute combination");
    }
  }

  collectDataSetForClusterAndLabel(labelType: string) {
    /**
     * This Method reformats the datapoint in a structure that is accepatable for the graph.
     * It does show the clusters in a different collor and the label in a different Form
     */
    const datasets: {
      label: string,
      data: [{ x: number, y: number, index: number }],
      pointStyle: [string],
      pointRadius: number
    }[] = [];
    const seenClusters: number[] = []
    this.myData.forEach((point: DynamicPoint) => {
      const clusterName: string = "Cluster " + point["cluster"];
      let form
      let currentLabel
      if (labelType == "actual") {
        currentLabel = point["actualLabel"]
      } else {
        currentLabel = point["predictedLabel"]
      }
      if (currentLabel == 1) {
        form = "triangle"
      } else {
        form = "circle"
      }
      if (!seenClusters.includes(point["cluster"])) {
        datasets.push({
          label: clusterName,
          data: [{x: point["x"], y: point["y"], index: point["index"]}],
          pointStyle: [form],
          pointRadius: 5,
        });
        seenClusters.push(point["cluster"]);
      } else {
        // find the dataset:
        for (const dataset of datasets) {

          if (dataset.label == clusterName) {
            dataset.data.push({x: point["x"], y: point["y"], index: point["index"]});
            dataset.pointStyle.push(form);

          }
        }
      }
    });
    return this.sortLabels(datasets)
  }

  collectDataSetForCluster() {
    /**
     * This Method reformats the datapoint in a structure that is accepatable for the graph.
     * It does show the clusters in a different
     */
    const datasets: {
      label: string,
      data: [{ x: number, y: number, index: number }],
      pointRadius: number
    }[] = [];
    const seenClusters: number[] = []
    this.myData.forEach((point: {
      x: number;
      y: number;
      cluster: number;
      actualLabel: number;
      predictedLabel: number;
      sensitive: number;
      index: number;
    }) => {
      const clusterName: string = "Cluster " + point.cluster;

      if (!seenClusters.includes(point.cluster)) {
        datasets.push({
          label: clusterName,
          data: [{x: point.x, y: point.y, index: point.index}],
          pointRadius: 5,
        });
        seenClusters.push(point.cluster);
      } else {
        // find the dataset:
        for (const dataset of datasets) {

          if (dataset.label == clusterName) {
            dataset.data.push({x: point.x, y: point.y, index: point.index});

          }
        }
      }
    });
    return this.sortLabels(datasets)
  }

  collectDataSetForLabel(labelType: string) {
    /**
     * This Method reformats the datapoint in a structure that is accepatable for the graph.
     * It does show the labels in a different color
     */
    const datasets: {
      label: string,
      data: [{ x: number, y: number, index: number }],
      pointRadius: number
    }[] = [];
    const seenLabels: number[] = []
    this.myData.forEach((point: {
      x: number;
      y: number;
      cluster: number;
      actualLabel: number;
      predictedLabel: number;
      sensitive: number;
      index: number;
    }) => {

      let currentLabel;
      if (labelType == "actual") {
        currentLabel = point.actualLabel;
      } else {
        currentLabel = point.predictedLabel;
      }
      const labelName: string = "Label " + currentLabel;
      if (!seenLabels.includes(currentLabel)) {
        datasets.push({
          label: labelName,
          data: [{x: point.x, y: point.y, index: point.index}],
          pointRadius: 5,
        });
        seenLabels.push(currentLabel);
      } else {
        // find the dataset:
        for (const dataset of datasets) {
          if (dataset.label == labelName) {
            dataset.data.push({x: point.x, y: point.y, index: point.index});

          }
        }
      }
    });
    return datasets
  }


  collectDataSetForSensitive2() {
    const datasets: {
      label: string,
      data: [{ x: number, y: number, index: number }],
      pointRadius: number
    }[] = [];
    const seenGroups: string[] = []
    this.myData.forEach((point: DynamicPoint) => {
      let labelName = ""
      for (let sensitiveAttribute of this.sensitiveAttributes) {
        labelName = labelName + sensitiveAttribute + point[sensitiveAttribute] + " "
      }
      if (!seenGroups.includes(labelName)) {
        datasets.push({
          label: labelName,
          data: [{x: point["x"], y: point["y"], index: point["index"]}],
          pointRadius: 5,
        })
        seenGroups.push(labelName)
      } else {
        // find the dataset:
        for (const dataset of datasets) {
          if (dataset.label == labelName) {
            dataset.data.push({x: point["x"], y: point["y"], index: point["index"]});

          }
        }
      }

    });
    return datasets


  }


  collectDataSetForSensitive() {
    /**
     * This Method reformats the datapoint in a structure that is accepatable for the graph.
     * It does show the labels in a different color
     */
    const datasets: {
      label: string,
      data: [{ x: number, y: number, index: number }],
      pointRadius: number
    }[] = [];
    const seenGroups: number[] = []
    this.myData.forEach((point: {
      x: number;
      y: number;
      cluster: number;
      actualLabel: number;
      predictedLabel: number;
      sensitive: number;
      index: number;
    }) => {
      const labelName: string = "Group " + point.sensitive;

      if (!seenGroups.includes(point.sensitive)) {
        datasets.push({
          label: labelName,
          data: [{x: point.x, y: point.y, index: point.index}],
          pointRadius: 5,
        });
        seenGroups.push(point.sensitive);
      } else {
        // find the dataset:
        for (const dataset of datasets) {
          if (dataset.label == labelName) {
            dataset.data.push({x: point.x, y: point.y, index: point.index});

          }
        }
      }
    });
    return datasets
  }

  collectDataSetForClusterAndSensitive() {
    /**
     * This Method reformats the datapoint in a structure that is accepatable for the graph.
     * It does show the clusters in a different collor and the label in a different Form
     */
    const datasets: {
      label: string,
      data: [{ x: number, y: number, index: number }],
      pointStyle: [string],
      pointRadius: number
    }[] = [];
    const seenClusters: number[] = []
    this.myData.forEach((point: DynamicPoint) => {
      const clusterName: string = "Cluster " + point["cluster"];
      let form
      if (this.sensitiveAttributes.length == 1) {
        if (point[this.sensitiveAttributes[0]] == 1) {
          form = "triangle"
        } else {
          form = "circle"
        }
      } else if (this.sensitiveAttributes.length == 2) {
        if (point[this.sensitiveAttributes[0]] == 1) {
          if (point[this.sensitiveAttributes[1]] == 1) {
            form = "rect"
          } else {
            form = "circle"
          }
        } else {
          if (point[this.sensitiveAttributes[1]] == 1) {
            form = "star"
          } else {
            form = "triangle"
          }
        }

      } else {
        // TODO More sens => only show favored / not favored
        form = "rect"
      }

      if (!seenClusters.includes(point["cluster"])) {
        datasets.push({
          label: clusterName,
          data: [{x: point["x"], y: point["y"], index: point["index"]}],
          pointStyle: [form],
          pointRadius: 5,
        });
        seenClusters.push(point["cluster"]);
      } else {
        // find the dataset:
        for (const dataset of datasets) {

          if (dataset.label == clusterName) {
            dataset.data.push({x: point["x"], y: point["y"], index: point["index"]});
            dataset.pointStyle.push(form);

          }
        }
      }
    });
    return this.sortLabels(datasets)
  }

  collectDataSetForClusterAndCorrect() {
    /**
     * This Method reformats the datapoint in a structure that is accepatable for the graph.
     * It does show the clusters in a different collor and the label in a different Form
     */
    const datasets: {
      label: string,
      data: [{ x: number, y: number, index: number }],
      pointStyle: [string],
      pointRadius: number
    }[] = [];
    const seenClusters: number[] = []
    this.myData.forEach((point: {
      x: number;
      y: number;
      cluster: number;
      actualLabel: number;
      predictedLabel: number;
      sensitive: number;
      index: number;
    }) => {
      const clusterName: string = "Cluster " + point.cluster;
      let form
      if (point.actualLabel == point.predictedLabel) {
        form = "triangle"
      } else {
        form = "circle"
      }
      if (!seenClusters.includes(point.cluster)) {
        datasets.push({
          label: clusterName,
          data: [{x: point.x, y: point.y, index: point.index}],
          pointStyle: [form],
          pointRadius: 5,
        });
        seenClusters.push(point.cluster);

      } else {
        // find the dataset:
        for (const dataset of datasets) {

          if (dataset.label == clusterName) {
            dataset.data.push({x: point.x, y: point.y, index: point.index});
            dataset.pointStyle.push(form);

          }
        }
      }
    });

    return this.sortLabels(datasets)
  }

  filterRunName(name: string){
    return name.split("-")[0]
  }

  collectDataSetForSensitiveAndCorrect() {
    /**
     * This Method reformats the datapoint in a structure that is accepatable for the graph.
     * It does show the clusters in a different collor and the label in a different Form
     */
    const datasets: {
      label: string,
      data: [{ x: number, y: number, index: number }],
      pointStyle: [string],
      pointRadius: number,
      backgroundColor: string,
      borderColor: string
    }[] = [];
    const seenGroups: string[] = []
    this.myData.forEach((point: DynamicPoint) => {

      let actual = point["actualLabel"]
      let predicted = point["predictedLabel"]
      let labelName;
      let borderColor;
      let backgroundColor;
      if (actual == predicted) {
        labelName = "correct";
        borderColor = "rgb(54, 162, 235)"
        backgroundColor = "rgba(54, 162, 235, 0.5)"
      } else {
        labelName = "incorrect";
        backgroundColor = "rgba(255, 99, 132, 0.5)"
        borderColor = "rgb(255, 99, 132)"
      }

      let form
      if (this.sensitiveAttributes.length == 1) {
        if (point[this.sensitiveAttributes[0]] == 1) {
          form = "triangle"
        } else {
          form = "circle"
        }
      } else if (this.sensitiveAttributes.length == 2) {
        if (point[this.sensitiveAttributes[0]] == 1) {
          if (point[this.sensitiveAttributes[1]] == 1) {
            form = "rect"
          } else {
            form = "circle"
          }
        } else {
          if (point[this.sensitiveAttributes[1]] == 1) {
            form = "star"
          } else {
            form = "triangle"
          }
        }

      } else {
        // TODO More sens => only show favored / not favored
        form = "triangle"
      }

      if (!seenGroups.includes(labelName)) {
        datasets.push({
          label: labelName,
          data: [{x: point["x"], y: point["y"], index: point["index"]}],
          pointStyle: [form],
          pointRadius: 5,
          backgroundColor: backgroundColor,
          borderColor: borderColor
        });
        seenGroups.push(labelName);
      } else {
        // find the dataset:
        for (const dataset of datasets) {
          if (dataset.label == labelName) {
            dataset.data.push({x: point["x"], y: point["y"], index: point["index"]});
            dataset.pointStyle.push(form);

          }
        }
      }
    });
    return datasets
  }

  collectDataSetForSensitiveAndLabel(labelType: string) {
    /**
     * This Method reformats the datapoint in a structure that is accepatable for the graph.
     * It does show the clusters in a different collor and the label in a different Form
     */
    const datasets: {
      label: string,
      data: [{ x: number, y: number, index: number }],
      pointStyle: [string],
      pointRadius: number
    }[] = [];
    const seenLabels: number[] = []
    this.myData.forEach((point: DynamicPoint) => {
      let currentLabel;
      if (labelType == "actual") {
        currentLabel = point["actualLabel"];
      } else {
        currentLabel = point["predictedLabel"];
      }

      const labelName: string = "Label " + currentLabel;
      let form
      if (this.sensitiveAttributes.length == 1) {
        if (point[this.sensitiveAttributes[0]] == 1) {
          form = "triangle"
        } else {
          form = "circle"
        }
      } else if (this.sensitiveAttributes.length == 2) {
        if (point[this.sensitiveAttributes[0]] == 1) {
          if (point[this.sensitiveAttributes[1]] == 1) {
            form = "rect"
          } else {
            form = "circle"
          }
        } else {
          if (point[this.sensitiveAttributes[1]] == 1) {
            form = "star"
          } else {
            form = "triangle"
          }
        }

      } else {
        // TODO More sens => only show favored / not favored
        form = "triangle"
      }
      if (!seenLabels.includes(currentLabel)) {
        datasets.push({
          label: labelName,
          data: [{x: point["x"], y: point["y"], index: point["index"]}],
          pointStyle: [form],
          pointRadius: 5,
        });
        seenLabels.push(currentLabel);
      } else {
        // find the dataset:
        for (const dataset of datasets) {

          if (dataset.label == labelName) {
            dataset.data.push({x: point["x"], y: point["y"], index: point["index"]});
            dataset.pointStyle.push(form);

          }
        }
      }
    });
    return datasets
  }

  collectDataSetForCorrect() {
    /**
     * This Method reformats the datapoint in a structure that is accepatable for the graph.
     * It does show the clusters in a different collor and the label in a different Form
     */
    const datasets: {
      label: string,
      data: [{ x: number, y: number, index: number }],
      pointRadius: number,
      borderColor: string,
      backgroundColor: string,
    }[] = [];
    const seenLabels: string[] = []
    this.myData.forEach((point: {
      x: number;
      y: number;
      cluster: number;
      actualLabel: number;
      predictedLabel: number;
      sensitive: number;
      index: number;
    }) => {
      let actual = point.actualLabel
      let predicted = point.predictedLabel
      let labelName;
      let borderColor;
      let backgroundColor;
      if (actual == predicted) {
        labelName = "correct";
        borderColor = "rgb(54, 162, 235)"
        backgroundColor = "rgba(54, 162, 235, 0.5)"
      } else {
        labelName = "incorrect";
        backgroundColor = "rgba(255, 99, 132, 0.5)"
        borderColor = "rgb(255, 99, 132)"
      }

      if (!seenLabels.includes(labelName)) {
        datasets.push({
          label: labelName,
          data: [{x: point.x, y: point.y, index: point.index}],
          pointRadius: 5,
          borderColor: borderColor,
          backgroundColor: backgroundColor,
        });
        seenLabels.push(labelName);
      } else {
        // find the dataset:
        for (const dataset of datasets) {

          if (dataset.label == labelName) {
            dataset.data.push({x: point.x, y: point.y, index: point.index});

          }
        }
      }
    });
    return datasets
  }

  resetZoomDataSetChart() {

    if (this.myChart) {
      this.myChart.resetZoom();

    }
  }


  activateButton(buttonNumber: number) {
    if (buttonNumber === 1) {
      this.button1Active = true;
      this.button2Active = false;
    } else {
      this.button1Active = false;
      this.button2Active = true;
    }
  }

  loading: boolean = false;

  startPrediction(number: number) {
    this.loading = true;
    let point;
    if (number == 1) {
      point = this.valuesPoint1;
    } else {
      point = this.valuesPoint2;
    }


    this.httpService.sendDataPoint(point, this.showModel, this.shownRun).subscribe((response: any) => {
      let entries = Object.entries(response);
      this.predictionRunResults = response.run_predictions[this.shownRun];
      // this.usedModel = this.predictionModelsResults["usedModel"];
      this.runKeys = Object.keys(this.predictionRunResults);


       // Raise a toast if any of the runs is a fair run because they cant predict a single point yet
      this.loading = false;
    });
  }


  modelResultVisible: { [key: string]: boolean } = {};

  showModelResults(name: string) {
    this.modelResultVisible[name] = !this.modelResultVisible[name];
  }

  protected readonly Object = Object;


  isUsedRow(runName: string, submodelName: string) {
    const isMatching = this.usedModels[runName] == submodelName;
    return isMatching;
  }

  takeScreenshot() {
    // @ts-ignore
    html2canvas(document.getElementById("chart-screenshot"), {logging: false})
        .then(canvas => this.startDownload(this.shownRun+ "chart_screenshot.png", canvas.toDataURL("image/png;base64")))
  }

  downloadCSV(){
    let csvString = "x, y, cluster, actualLabel, predictedLabel, index"
    for (let sensitiveAttribute of this.sensitiveAttributes) {
        csvString = csvString + ", "+ sensitiveAttribute;
    }
    csvString = csvString + "\n";
    this.myData.forEach((point: DynamicPoint
    ) => {
      csvString += `${point["x"]},${point["y"]},${point["cluster"]},${point["actualLabel"]},${point["predictedLabel"]},${point["index"]}`;
      for (let sensitiveAttribute of this.sensitiveAttributes) {
        csvString = csvString + ", "+ point[sensitiveAttribute];
      }
      csvString = csvString + "\n"
    })
    // Erstelle einen Blob aus dem gesamten CSV-String
    const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
    this.startDownload(this.shownRun + "_values.csv", URL.createObjectURL(blob));
  }

  // @ts-ignore
  startDownload(name, href) {
    let htmlElement = document.createElement("a");
    htmlElement.href = href;
    htmlElement.download = name;
    htmlElement.click();
  }

  sortLabels(datasets: any[]) {
    return datasets.sort((a, b) => {
      if (a["label"] < b["label"]) {
        return -1
      }
      if (a["label"] > b["label"]) {
        return 1
      }
      return 0
    })
  }


  getCounterfactual() {
    if (this.shownRun) {
      this.httpService.getCounterFactual(this.indexPoint1, this.shownRun, this.showModel).subscribe((response: any) => {
        let index = response.index
        let actualLabel = response.actualLabel
        let predictedLabel = response.predictedLabel
        let cluster = response.cluster
        this.highLightCounterFactual(index)
      });
    }
  }

  oldDatasetIndex: number = -1
  oldIndex: number = -1

  highlightPoint(datasetIndex: number, index: number) {
    if (this.myChart) {

      // save the points
      this.oldDatasetIndex = datasetIndex
      this.oldIndex = index

      if (datasetIndex < 0 || datasetIndex >= this.myChart.data.datasets.length) {
        return;
      }
      // reset the old highlights
      this.myChart.data.datasets.forEach((dataset: any, datasetIndex: number) => {
        dataset.pointRadius = dataset.data.map(() => 5);
      });


      const dataset = this.myChart.data.datasets[datasetIndex] as any;
      dataset.pointRadius = dataset.data.map(() => 5);
      dataset.pointRadius[index] = 10;
      this.oldDatasetIndex = datasetIndex
      this.myChart.update();
    }
  }

  oldCounerFactualIndex = -1
  oldCounterFactialDataset = -1

  highLightCounterFactual(targetId: number): void {
    if (!this.myChart) return;
    this.myChart.data.datasets.forEach((dataset: any, datasetIndex: number) => {
      dataset.pointRadius = dataset.data.map(() => 5);
      dataset.data.forEach((point: any, pointIndex: number) => {
        if (point.index === targetId || (datasetIndex == this.oldDatasetIndex && pointIndex == this.oldIndex)) {
          dataset.pointRadius[pointIndex] = 10;
          this.oldCounterFactialDataset = datasetIndex
          this.oldCounerFactualIndex = pointIndex
        }
      });
    });

    this.myChart.update();
  }




  Colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'
  ];
  showModel: any;


}

