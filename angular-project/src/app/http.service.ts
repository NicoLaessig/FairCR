import { Injectable } from '@angular/core';
import {HttpClient, HttpParams} from "@angular/common/http";

@Injectable({
  providedIn: 'root'
})
export class HttpService {
  // the url of the server
  private url = "http://127.0.0.1:5000";
  private urlDatasetAnalyzer = this.url + "/dataset-analyzer";
  private urlRunAnalyzer = this.url + "/run-analyzer"
  private urlRecommendations = this.url + "/recommendations"

  constructor(private http: HttpClient) { }

  /**
   * This Method sends the data from the input fields to the server as http post request (Because the state of the server changes)
   * It returns an Observable object
   * @param data
   */
  sendDatasetInput(data: any){
    return this.http.post(this.urlDatasetAnalyzer + "/inputdata", data);
  }

  /**
   * This Methods sends 2 attribute names to the server as http get request
   * @param attribute1
   * @param attribute2
   */
  sendDatasetAttributes(attribute1: string, attribute2: string){
    const urlWithParams = this.urlDatasetAnalyzer + "/inputdata/attributes" + `?attribute1=${attribute1}&attribute2=${attribute2}`;
    return this.http.get(urlWithParams);
  }

  /**
   * This Method sends a single datapoint in tsne format from the chart to the server as http get (TODO change) request
   * @param datapoint
   */
  sendDatasetTSNEPoint(dataset: number, index: number, coordinate: object){
    const urlWithParams = this.urlDatasetAnalyzer + "/datapoint" + `?dataset=${dataset}&index=${index}&coordinate=${coordinate}`;
    return this.http.get(urlWithParams);
  }

  /**
   * This Method sends two datapoints to the server as http post request
   */
  sendDatasetPoints(datapoint1: number[] , datapoint2: number[]){
    const urlWithParams = this.urlDatasetAnalyzer + "/datapoint/similiarity" + `?attribute1=${datapoint1}&attribute2=${datapoint2}`;
    return this.http.get(urlWithParams);
  }


  /*         HERE ARE THE METHODS FOR THE RUN ANALYZER  TODO: split it in 2 services          */

  initRuns(data: any){
    return this.http.post(this.urlRunAnalyzer + "/init-runs", data);
  }

  getAllAttributes(runName: string){
    const urlWithParams = this.urlRunAnalyzer + "/inputdata/getattributes" + `?runName=${runName}`;
    return this.http.get(urlWithParams)
  }

  trainRuns(data:any){
    return this.http.post(this.urlRunAnalyzer + "/train-runs", data);
  }
  sendAttributes(runName: string | null, modelName: string | null, attribute1: string, attribute2: string){
    const urlWithParams = this.urlRunAnalyzer + "/inputdata/attributes" + `?attribute1=${attribute1}&attribute2=${attribute2}&runName=${runName}&modelName=${modelName}`;
    return this.http.get(urlWithParams);
  }

  sendOriginalIndex(index: number, modelName: string, runName: string){
    const urlWithParams = this.urlRunAnalyzer + "/inputdata/index" + `?index=${index}&runName=${runName}&modelName=${modelName}`;
    return this.http.get(urlWithParams)
 }

  sendDataPoint(datapoint: any, shownModel: any, shownRun: any){
    let data  = {
      "modelName": shownModel,
      "runName": shownRun,
      "datapoint": datapoint
    };
    return this.http.post(this.urlRunAnalyzer + "/inputdata/predict", data);
 }


  getTrainedRuns() {
    const url = this.urlRunAnalyzer + "/inputdata/trainedruns"
    return this.http.get(url)
  }


  getCounterFactual(datapointIndex: number, runName: string, modelName: string){
    const urlWithParams = this.urlRunAnalyzer + "/inputdata/counterfactual" + `?index=${datapointIndex}&runName=${runName}&modelName=${modelName}`;
    return this.http.get(urlWithParams)
  }

  getClusterInformation(clusterNumber: number, runName: string){
    const urlWithParams = this.urlRunAnalyzer + "/inputdata/clusterInformation" + `?cluster=${clusterNumber}&runName=${runName}`;
    return this.http.get(urlWithParams)
  }

  getModelCombination(clusterNumber: number, runName: string, weight: number, metric: string){
    const urlWithParams = this.urlRunAnalyzer + "/inputdata/bestModelCombination" + `?cluster=${clusterNumber}&runName=${runName}&weight=${weight}&metric=${metric}`;
    return this.http.get(urlWithParams)
  }

  getSpecificRecommendations(runName: string){
    const urlWithParams = this.urlRunAnalyzer + "/specificrecommendations" + `?runName=${runName}`;
    return this.http.get(urlWithParams)
  }

  getGeneralRecommendations(metric: string, tuning: boolean, lam: string,
                            local_lam: string, datasize: string, mem: string, time: string){
    const urlWithParams = this.urlRecommendations + "/generalrecommendations" + `?metric=${metric}&tuning=${tuning}&lambda=${lam}&localLambda=${local_lam}&datasize=${datasize}&mem=${mem}&time=${time}`;
    return this.http.get(urlWithParams)
  }

  // Service to get the config out of the assets folder
  getConfigs(){
    return this.http.get('/assets/configs/params.json')
  }

  // Service to store a new dataset in the directory

  postDataset(file: File){
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(this.urlRunAnalyzer + "/uploadDataset", formData);
  }

  uploadDataset(formData: FormData){
    return this.http.post(this.urlRunAnalyzer + "/uploadDataset", formData);
  }
  getDatasets(){
    return this.http.get(this.urlRunAnalyzer + "/getDatasetNames")
  }


}
