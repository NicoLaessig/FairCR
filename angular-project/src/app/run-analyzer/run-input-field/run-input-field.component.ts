import {Component, OnInit} from '@angular/core';
import {HttpService} from "../../http.service";
import {SharedDataService} from "../../shared-data.service";
import {ToastrService} from "ngx-toastr";


interface Runs {
    [runName: string]: {
        [innerKey: string]: string;
    };
}

@Component({
    selector: 'app-run-input-field',
    templateUrl: './run-input-field.component.html',
    styleUrls: ['./run-input-field.component.scss']
})
export class RunInputFieldComponent implements OnInit {
    // Every trained run in the server
    totalTrainedRuns: Runs = {};
    dataIsLoaded: boolean = false;
    // The runs configs that we selected that are already trained
    trainedRuns: string[] = [];
    notTrainedRuns: string[] = [];
    currentShownRun = 0;
    // store for each run name if it should be retrained
    train: { [key: string]: boolean } = {};
    // data containing the different models and their parameters
    // configs: any = [];
    // modelTypes: string[] = []
    // data containing the selected model types and how many different configurations use them
    // data containg the names of the different datasets
    datasets: any = []
    datasetDetails: any
    Runs = [{
        name: 'Run 1',
        dataset: 'communities',
        testsize: '0.3',
        index: 'index',
        sensitive: 'race',
        favored: '1',
        label: 'crime',
        metric: 'demographic_parity',
        lamda: '0.5',
        localLamda: '0.5',
        randomstate: '-1',
        models: [],
        numberOfRuns: '1',
        hyperparameterTuning: false,
        runTime: '0.0',
        memory: '0.8'
    }];

    newDataset: any
    newIndexName: string = ""
    newLabelName: string = ""
    newSensitiveName: string = ""
    newFavoredName: string = ""

    checkboxStatuses: { [key: string]: boolean } = {};

    constructor(private httpService: HttpService, private sharedDataService: SharedDataService, private toastr: ToastrService) {
    }

    ngOnInit() {
        // this.httpService.getConfigs().subscribe(data => {
        //     this.configs = data;
        //     this.modelTypes = Object.keys(this.configs)
        // });
        this.httpService.getDatasets().subscribe((data: any) => {
          this.datasets = data["datasetNames"];
          this.datasetDetails = data["datasetDetails"]
        })
        this.dataIsLoaded = true;

    }


    isModelSelected(model: string){
      let selected = false;
      for(let selectedModel of this.Runs[this.currentShownRun].models){
        if(model == selectedModel){
          selected =  true;
        }
      }
      return selected;
    }


    onModelChange(checkboxName: string, event: Event) {
        const isChecked = (event.target as HTMLInputElement).checked;
        // remove this model from the modelList
        this.Runs[this.currentShownRun].models = this.Runs[this.currentShownRun].models.filter(model => model != checkboxName);
        if(isChecked){
          // if event = true add it again
          // @ts-ignore
          this.Runs[this.currentShownRun].models.push(checkboxName);
        }

        this.checkboxStatuses[checkboxName] = isChecked
    }


    addTrainedRun(name: string, config: any) {
        config["name"] = name
        this.Runs.push(config)
        this.currentShownRun = this.Runs.length - 1;
        this.checkboxStatuses[config["model"]] = true;
    }


    /**
     * This function is called when pressing the submit Button.
     * It takes the data from the different runs and sends it to the backend and waits for the response.
     *
     * !!! this can take very long !!!
     */
    submitData() {
        // Daten aus dem Eingabeformular abrufen

        let data = this.Runs;

        if (data.length > 0) {
            // Check which runs has to be trained and display warning
            this.httpService.initRuns(data).subscribe(
                (response: any) => {
                    this.notTrainedRuns = response["notTrainedRuns"];
                    this.trainedRuns = response["trainedRuns"];
                    // init

                    for (let run of this.trainedRuns) {
                        this.train[run] = false;
                    }
                    if (this.trainedRuns.length > 0) {
                        const modelDiv = document.getElementById("myModal2")
                        if (modelDiv != null) {
                            modelDiv.style.display = 'block';
                        }
                    } else {
                        this.trainRuns();
                    }
                },
                error => {
                    console.error("Fehler beim Senden der Daten an den Server:", error);
                }
            );
        }

    }

    trainRuns() {

        let data = {"runConfigs": this.Runs, "retrainInformations": this.train};

        // get the runs that have to be trained
        let todoRuns: string[] = this.notTrainedRuns;
        for (let run in this.train) {
            if (this.train[run] == true) {
                todoRuns.push(run);
            }
        }

        if (this.notTrainedRuns.length > 0) {
            this.toastr.warning("The following are getting trained: " + todoRuns.join(',') + "", "Long Runtime!");
            // train the runs
        }

        // Trainiere die Runs
        this.httpService.trainRuns(data).subscribe(
            (response: any) => {
                this.sharedDataService.updateMetricData(response);
                this.toastr.success("All Runs are trained", "Success");
            },
            error => {
                this.toastr.error("An error occurred while sending data to the server.", "Error");
            }
        );

        this.closeModel2()
    }

    changeShownRun(index: number) {
        this.currentShownRun = index;
    }

    // deleteRun(index: number) {
    //   if (index >= 0 && index < this.runs.length && this.runs.length > 1) {
    //     this.runs.splice(index, 1);
    //     this.currentShownRun = 0;
    //   }
    // }

    isShownRun(index: number) {
        return this.currentShownRun === index;
    }

    getTrainedRuns() {
        console.log("hier")
        this.httpService.getTrainedRuns().subscribe(
            (response: any) => {
                this.totalTrainedRuns = response;

                const modelDiv = document.getElementById("myModal")
                if (modelDiv != null) {
                    modelDiv.style.display = 'block';
                }

            },
            error => {
                console.error("Fehler beim Empfangen der Daten an den Server:", error);
                this.toastr.error("An error occurred while getting data to the server.", "Error");
            }
        );
    }

    closeModel() {
        const modelDiv = document.getElementById("myModal")
        if (modelDiv != null) {
            modelDiv.style.display = 'none';
        }
    }

    getOuterKeys(): string[] {
        return Object.keys(this.totalTrainedRuns);
    }

    getInnerKeys(): string[] {
        const outerKeys = this.getOuterKeys();
        if (outerKeys.length > 0) {
            const innerObject = this.totalTrainedRuns[outerKeys[0]];

            if (innerObject) {
                return Object.keys(innerObject);
            }
        }
        return [];
    }

    closeModel2() {
        const modelDiv = document.getElementById("myModal2")
        if (modelDiv != null) {
            modelDiv.style.display = 'none';
        }
    }


    duplicateToastShown: boolean = false;

// checkForDuplicateRuns(): boolean {
//   for (let i = 0; i < this.runs.length; i++) {
//     for (let j = i + 1; j < this.runs.length; j++) {
//       const runI = JSON.stringify({ ...this.runs[i], name: '', numberOfRuns: '' });
//       const runJ = JSON.stringify({ ...this.runs[j], name: '', numberOfRuns: '' });
//       if (runI === runJ) {
//         if (!this.duplicateToastShown) {
//           this.toastr.warning("Multiple Runs have the same configurations. Use the Count option if you want to train mulitple similiar runs", "Same Run configurations");
//           this.duplicateToastShown = true;
//         }
//         return true;
//       }
//     }
//   }
//   this.duplicateToastShown = false;
//   return false;
// }


    closeInformationModalInput() {
        const modelDiv = document.getElementById("informationModalInput")
        if (modelDiv != null) {
            modelDiv.style.display = 'none';
        }
    }

    showInformationModel() {
        const modelDiv = document.getElementById("informationModalInput")
        if (modelDiv != null) {
            modelDiv.style.display = 'block';
        }
    }

    onDataSetChange() {
        let dataset = this.Runs[this.currentShownRun].dataset
        this.Runs[this.currentShownRun].index = this.datasetDetails[dataset]["index"]
        this.Runs[this.currentShownRun].label = this.datasetDetails[dataset]["label"]
        this.Runs[this.currentShownRun].sensitive = this.datasetDetails[dataset]["protectedAttributes"]
        this.Runs[this.currentShownRun].favored = this.datasetDetails[dataset]["FavoredAttributes"]
    }

    isArray(value: any): boolean {
      return Array.isArray(value);
    }

    showDatasetUploadModel() {
        const modelDiv = document.getElementById("myModal3")
        if (modelDiv != null) {
            modelDiv.style.display = 'block';
        }
    }

    closeModel3() {
        const modelDiv = document.getElementById("myModal3")
        if (modelDiv != null) {
            modelDiv.style.display = 'none';
        }
    }

    onFileSelected(event: Event): void {
        const element = event.target as HTMLInputElement;
        const file: File | null = element.files ? element.files[0] : null;

        if (file) {
            this.httpService.postDataset(file).subscribe((data: any) => {
                this.datasets = data["datasetNames"];
                this.datasetDetails = data["datasetDetails"]
                this.toastr.success("uploaded new datafile", "Success")
            });
        } else {
            console.log('No file selected.');
        }
    }

  copyRun(index: number) {
    const copyRun = JSON.parse(JSON.stringify(this.Runs[index]))
    copyRun.name = 'Run ' + (this.Runs.length + 1)
    this.Runs.push(copyRun)
    this.currentShownRun = this.Runs.length - 1;
  }

  deleteRun(index: number) {
    if (index >= 0 && index < this.Runs.length && this.Runs.length > 1) {
      this.Runs.splice(index, 1);
      this.currentShownRun = 0;
    }
  }

  addRun() {
    this.Runs.push({
        name: 'Run' + (this.Runs.length + 1),
        dataset: 'communities',
        testsize: '0.3',
        index: 'index',
        sensitive: 'race',
        favored: '1',
        label: 'crime',
        metric: 'demographic_parity',
        lamda: '0.5',
        localLamda: '0.5',
        randomstate: '-1',
        models: [],
        numberOfRuns: '1',
        hyperparameterTuning: false,
        runTime: '0.0',
        memory: '0.8'
    })
    this.currentShownRun = this.Runs.length - 1;
  }

  onnewDatasetFileSelected(event: Event) {
    const target = event.target as HTMLInputElement;
    this.newDataset = (target.files as FileList)[0];
  }

  uploadDataset() {
    if(this.newDataset){
      const formData = new FormData()
      formData.append('file', this.newDataset, this.newDataset.name)
      formData.append('index', this.newIndexName);
      formData.append('label', this.newLabelName);
      formData.append('sensitive', this.newSensitiveName);
      formData.append('favored', this.newFavoredName);
      this.httpService.uploadDataset(formData).subscribe((data: any) => {
        this.datasets = data["datasetNames"];
        this.datasetDetails = data["datasetDetails"]
        this.toastr.success("uploaded new datafile", "Success")
      });
    }
  }
}
