import {Component, Input, OnInit, OnChanges} from '@angular/core';
import {Chart, TooltipModel} from "chart.js";
import html2canvas from "html2canvas";

interface DataPoint {
  x: number;
  y: number;
  label: string;
  modelName: string;
}


function makeColorSubtle(color: string): string {
  color = color.replace('#', '');
  const r = parseInt(color.slice(0, 2), 16);
  const g = parseInt(color.slice(2, 4), 16);
  const b = parseInt(color.slice(4, 6), 16);
  const brightnessFactor = 0.7; // 70% der Helligkeit behalten
  const saturationFactor = 0.5; // 50% der Sättigung behalten

  const newR = Math.floor(r * brightnessFactor);
  const newG = Math.floor(g * brightnessFactor);
  const newB = Math.floor(b * brightnessFactor);

  const maxComponent = Math.max(newR, newG, newB);
  const minComponent = Math.min(newR, newG, newB);
  const avgComponent = (maxComponent + minComponent) / 2;
  const delta = maxComponent - minComponent;
  const newS = Math.floor(avgComponent > 128 ? delta / (510 - maxComponent - minComponent) : delta / (maxComponent + minComponent));

  const newColor = `#${newR.toString(16).padStart(2, '0')}${newG.toString(16).padStart(2, '0')}${newB.toString(16).padStart(2, '0')}`;

  return newColor;
}

// Funktion zur Anpassung der Farbe für nicht "FALCC" Subnamen
function adjustColorForSubname(hexColor: any, subName: any) {
  if (subName.startsWith("FALCC")) {
    // Keine Anpassung für "FALCC" Subnamen, volle Opazität
    return hexColor;
  } else {
    // Reduzierte Opazität für weniger auffällige Farben
    console.log("old color: " + hexColor + "new color: " + makeColorSubtle(hexColor))
    return makeColorSubtle(hexColor);
  }
}

@Component({
  selector: 'app-metric-scatter-accordion-item',
  templateUrl: './metric-scatter-accordion-item.component.html',
  styleUrls: ['./metric-scatter-accordion-item.component.scss']
})


export class MetricScatterAccordionItemComponent implements OnInit, OnChanges {


  myChart: Chart | null = null;
  xMetric: string = "error_rate";
  yMetric: string = "demographic_parity";
  @Input() metrics: string[] = [];
  @Input() totalMetricData: any | null = null;
  @Input() allRuns: string[] | null = null;
  @Input() modelNames: any;
  displayType: string = "model"
  allModels: string[] = []
  shownRun: any
  shownModel: any
  showAvg: boolean = false;
  showSubModels: boolean = false;

  // Todo more forms
  pointStyle: string[] = ["circle", "triangle", "rect", "cross"]


  ngOnInit() {
    if (this.allRuns) {
      for (let run of this.allRuns) {
        for (let model in this.totalMetricData[0][run]) {
          let shortenModelName = model
          const suffix = "_tuned"
          if (model.endsWith("_tuned")) {
            shortenModelName = shortenModelName.slice(0, -suffix.length)
          }
          if (!this.allModels.includes(shortenModelName)) {
            this.allModels.push(shortenModelName)
          }
        }
      }
      this.shownRun = this.allRuns[0]
      this.shownModel = this.allModels[0]
      this.createMetricScatterChart(false);
    }
  }

  ngOnChanges() {
    this.createMetricScatterChart(false);
  }

  calculateDataForModel(wantAvg: boolean) {
    let dataIndex: number
    if (wantAvg) {
      dataIndex = 0
    } else {
      dataIndex = 1
    }
    let index = 0
    let datasets = []
    for (let run in this.totalMetricData[dataIndex]) {
      let hasModel = false
      let data = []
      for (let model in this.totalMetricData[dataIndex][run]) {
        if (model.startsWith(this.shownModel)) {
          hasModel = true;
          data.push({
            x: this.totalMetricData[dataIndex][run][model][this.xMetric],
            y: this.totalMetricData[dataIndex][run][model][this.yMetric],
            label: model
          })
        }
      }
      if (hasModel) {
        index++;
        datasets.push({
          label: run,
          data: data,
          pointRadius: 6.5,
          backgroundColor: colors[index],
          borderColor: borderColors[index]
        })
      }
    }
    return datasets
  }


  calculateDataForRun(wantAvg: boolean) {
    let dataIndex: number
    if (wantAvg) {
      dataIndex = 0
    } else {
      dataIndex = 1
    }
    // mapping every model to a list of points
    let models: any = {}
    for (let model in this.totalMetricData[dataIndex][this.shownRun]) {
      // remove number
      let model_name = model.replace(/\d+$/, '');
      if (models[model_name] != null) {
        models[model_name].push({
          x: this.totalMetricData[dataIndex][this.shownRun][model][this.xMetric],
          y: this.totalMetricData[dataIndex][this.shownRun][model][this.yMetric],
          label: model
        })
      } else {
        models[model_name] = [{
          x: this.totalMetricData[dataIndex][this.shownRun][model][this.xMetric],
          y: this.totalMetricData[dataIndex][this.shownRun][model][this.yMetric],
          label: model
        }]
      }
    }
    let datasets = []
    let index = 0
    for (let model in models) {
      datasets.push({
        label: model,
        data: models[model],
        pointRadius: 6.5,
        backgroundColor: colors[index],
        borderColor: borderColors[index]
      })
      index++;
    }
    return datasets
  }

    calculateDataForBoth(wantAvg: boolean) {
    let dataIndex: number
    if (wantAvg) {
      dataIndex = 0
    } else {
      dataIndex = 1
    }
    // mapping every model to a list of points
    let models: any = {}
    let formOfModels: any = {}
    let formIndex = 0
    for (let run in this.totalMetricData[dataIndex]){
      this.runToForm[run] = this.pointStyle[formIndex]
      for (let model in this.totalMetricData[dataIndex][run]) {
        // remove number
        let model_name = model.replace(/\d+$/, '');
        if (models[model_name] != null) {

          models[model_name].push({
            x: this.totalMetricData[dataIndex][run][model][this.xMetric],
            y: this.totalMetricData[dataIndex][run][model][this.yMetric],
            label: run + " - " + model
          })
          formOfModels[model_name].push(this.pointStyle[formIndex])

        } else {
          models[model_name] = [{
            x: this.totalMetricData[dataIndex][run][model][this.xMetric],
            y: this.totalMetricData[dataIndex][run][model][this.yMetric],
            label: run + " - " + model
          }]
          formOfModels[model_name] = [this.pointStyle[formIndex]]
        }
      }
      formIndex++;
      if(formIndex > this.pointStyle.length){
        formIndex == 0;
      }
    }
    let datasets = []
    let index = 0
    for (let model in models) {

      datasets.push({
        label: model,
        data: models[model],
        pointRadius: 6.5,
        backgroundColor: colors[index],
        borderColor: borderColors[index],
        pointStyle: formOfModels[model]
      })
      index++;
    }
    console.log("run to form")
    console.log(this.runToForm)
    return datasets
  }


  showMetricScatter() {
    const configMetricChart: any = {
      type: 'scatter',
      data: {
        datasets: []
      },
      options: {
        interaction: {
          mode: 'nearest'
        },
        scales: {
          x: {
            type: 'linear',
            position: 'bottom',
            title: {
              display: true,
              text: this.xMetric,
              font: {
                size: 18
              }
            },
            ticks: {
              font: {
                size: 14
              }
            }
          },
          y: {
            type: 'linear',
            position: 'left',
            title: {
              display: true,
              text: this.yMetric,
              font: {
                size: 18
              }
            },
            ticks: {
              font: {
                size: 14
              }
            }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              footer: function (tooltipItems: TooltipModel<'scatter'>[]) {

                const x = (tooltipItems[0] as any).raw.x;
                const y = (tooltipItems[0] as any).raw.y;
                const label = (tooltipItems[0] as any).raw.label;
                const dataset = (tooltipItems[0] as any).dataset;
                const datapoints = dataset["data"]
                const length = datapoints.length
                let similiarPoints = []
                similiarPoints[0] = label

                for (let index = 0; index < length; index++) {
                  let newPoint = datapoints[index]
                  if (newPoint.label != label) {
                    if (newPoint.x == x && newPoint.y == y) {
                      similiarPoints.push(newPoint.label)
                    }
                  }

                }
                return similiarPoints


              },
            }
          },
          zoom: {
            zoom: {
              wheel: {
                enabled: true
              },
              pinch: {
                enabled: true
              },
              mode: 'xy',
            }
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
    this.myChart = new Chart('analyzer-scatter-chart', configMetricChart);
  }

  resetZoomMetricScatterChart() {

    if (this.myChart) {
      this.myChart.resetZoom();
    }
  }

  filterRunName(name: string) {
    return name.split("-")[0]
  }

  createMetricScatterChart(showSubModels: boolean) {
    this.showMetricScatter();
    this.setMetricScatterDataPoints();
  }

  setMetricScatterDataPoints() {
    if (this.myChart && this.totalMetricData[0] != null) {
      let datasets
      if (this.displayType == "run") {
        datasets = this.calculateDataForRun(this.showAvg);
      } else if (this.displayType == "model") {
        datasets = this.calculateDataForModel(this.showAvg);
      } else if (this.displayType == "both"){
        console.log("hiersss")
        datasets = this.calculateDataForBoth(this.showAvg);
      }else{
        console.log("wrong type")
      }
      // @ts-ignore
      this.myChart.config.data.datasets = datasets
      // @ts-ignore
      this.myChart.config.options.scales.y.title.text = this.yMetric
      // @ts-ignore
      this.myChart.config.options.scales.x.title.text = this.xMetric
      this.myChart.update();
      this.resetZoomMetricScatterChart()
    }

  }

  onSwitchChange() {
    this.setMetricScatterDataPoints();
  }

  combinedMetric = ["accuracy"]
  weights = [1]
  metricName = ""

  addMetric() {
    this.combinedMetric.push("");
    this.weights.push();
  }

  openMetricCreaterModal() {
    const modelDiv = document.getElementById("metricCreaterModal")
    if (modelDiv != null) {
      modelDiv.style.display = 'block';
    }
  }

  closeMetricCombinerModel() {
    const modelDiv = document.getElementById("metricCreaterModal")
    if (modelDiv != null) {
      modelDiv.style.display = 'none';
    }
  }

  removeMetric(index: number) {
    if (this.combinedMetric.length > 1) {
      this.combinedMetric.splice(index, 1)
      this.weights.splice(index, 1)
    }
  }

  addNewMetric() {
    let totalWeight = 0
    for (let index = 0; index < this.weights.length; index++) {
      totalWeight = totalWeight + Number(this.weights[index]);
    }
    this.metrics.push(this.metricName)
    for (let run in this.totalMetricData[0]) {
      for (let subRun in this.totalMetricData[0][run]) {
        let totalValue = 0;
        for (let metric in this.totalMetricData[0][run][subRun]) {
          let index = this.combinedMetric.indexOf(metric);
          if (index !== -1) {
            let weight = this.weights[index]
            let value = this.totalMetricData[0][run][subRun][metric]
            totalValue = totalValue + (weight / totalWeight) * value;
          }
        }
        this.totalMetricData[0][run][subRun][this.metricName] = totalValue
      }
    }
    for (let run in this.totalMetricData[1]) {
      for (let subRun in this.totalMetricData[1][run]) {
        let totalValue = 0;
        for (let metric in this.totalMetricData[1][run][subRun]) {
          let index = this.combinedMetric.indexOf(metric);
          if (index !== -1) {
            let weight = this.weights[index]
            let value = this.totalMetricData[1][run][subRun][metric]
            totalValue = totalValue + (weight / totalWeight) * value;
          }
        }
        this.totalMetricData[1][run][subRun][this.metricName] = totalValue
      }
    }
    this.closeMetricCombinerModel();
  }

  takeScreenshot() {
    // @ts-ignore
    html2canvas(document.getElementById("metric-chart-screenshot"), {logging: false})
      .then(canvas => this.startDownload("MetricScatterChart.png", canvas.toDataURL("image/png;base64")))
  }

  downloadCSV() {
    let csvString = "runName, modelName"
    csvString = csvString + ", " + this.xMetric
    csvString = csvString + ", " + this.yMetric
    csvString = csvString + "\n"
    for (const runName in this.totalMetricData[0]) {
      for (const modelName in this.totalMetricData[0][runName]) {
        csvString = csvString + runName + ", " + modelName
        csvString = csvString + ", " + this.totalMetricData[0][runName][modelName][this.xMetric]
        csvString = csvString + ", " + this.totalMetricData[0][runName][modelName][this.yMetric]
        csvString = csvString + "\n"
      }
    }
    const blob = new Blob([csvString], {type: 'text/csv;charset=utf-8;'});
    this.startDownload("AverageResults", URL.createObjectURL(blob));
  }

  // @ts-ignore
  startDownload(name, href) {
    let htmlElement = document.createElement("a");
    htmlElement.href = href;
    htmlElement.download = name;
    htmlElement.click();
  }

  runToForm: any = {}
  getShapeClass(form: string): string {
    const shapeClasses: any = {
      circle: 'shape-circle',
      triangle: 'shape-triangle',
      rect: 'shape-rect',
      cross: 'shape-cross'
      // Ergänze weitere Formen hier
    };
    return shapeClasses[form] || 'shape-rect'; // Standardform
  }

}




const colors = [
  'rgba(255, 99, 132, 0.5)',   // Rot, halbtransparent
  'rgba(54, 162, 235, 0.5)',   // Blau, halbtransparent
  'rgba(255, 206, 86, 0.5)',   // Gelb, halbtransparent
  'rgba(75, 192, 192, 0.5)',   // Türkis, halbtransparent
  'rgba(153, 102, 255, 0.5)',  // Violett, halbtransparent
  'rgba(255, 159, 64, 0.5)',   // Orange, halbtransparent
  'rgba(199, 199, 199, 0.5)',  // Grau, halbtransparent
  'rgba(233, 30, 99, 0.5)',    // Pink, halbtransparent
  'rgba(0, 188, 212, 0.5)',    // Cyan, halbtransparent
  'rgba(255, 87, 34, 0.5)',    // Tiefes Orange, halbtransparent
  'rgba(155, 89, 182, 0.5)',   // Amethyst, halbtransparent
  'rgba(26, 188, 156, 0.5)',   // Türkis Grün, halbtransparent
  'rgba(46, 204, 113, 0.5)',   // Smaragdgrün, halbtransparent
  'rgba(52, 73, 94, 0.5)',     // Nassasphalt, halbtransparent
  'rgba(241, 196, 15, 0.5)',   // Sonnenblume, halbtransparent
  'rgba(230, 126, 34, 0.5)',   // Kürbis, halbtransparent
  'rgba(231, 76, 60, 0.5)',    // Granatapfel, halbtransparent
  'rgba(236, 240, 241, 0.5)',  // Wolken, halbtransparent
  'rgba(149, 165, 166, 0.5)',  // Beton, halbtransparent
  'rgba(243, 156, 18, 0.5)'    // Orange Squash, halbtransparent
];
const borderColors = [
  'rgba(255, 99, 132, 1)',   // Rot, halbtransparent
  'rgba(54, 162, 235, 1)',   // Blau, halbtransparent
  'rgba(255, 206, 86, 1)',   // Gelb, halbtransparent
  'rgba(75, 192, 192, 1)',   // Türkis, halbtransparent
  'rgba(153, 102, 255, 1)',  // Violett, halbtransparent
  'rgba(255, 159, 64, 1)',   // Orange, halbtransparent
  'rgba(199, 199, 199, 1)',  // Grau, halbtransparent
  'rgba(233, 30, 99, 1)',    // Pink, halbtransparent
  'rgba(0, 188, 212, 1)',    // Cyan, halbtransparent
  'rgba(255, 87, 34, 1)',    // Tiefes Orange, halbtransparent
  'rgba(155, 89, 182, 1)',   // Amethyst, halbtransparent
  'rgba(26, 188, 156, 1)',   // Türkis Grün, halbtransparent
  'rgba(46, 204, 113, 1)',   // Smaragdgrün, halbtransparent
  'rgba(52, 73, 94, 1)',     // Nassasphalt, halbtransparent
  'rgba(241, 196, 15, 1)',   // Sonnenblume, halbtransparent
  'rgba(230, 126, 34, 1)',   // Kürbis, halbtransparent
  'rgba(231, 76, 60, 1)',    // Granatapfel, halbtransparent
  'rgba(236, 240, 241, 1)',  // Wolken, halbtransparent
  'rgba(149, 165, 166, 1)',  // Beton, halbtransparent
  'rgba(243, 156, 18, 1)'    // Orange Squash, halbtransparent
];
