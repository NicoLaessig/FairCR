import { NgModule } from '@angular/core';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import {HeaderComponent} from "./header/header.component";
import {BrowserModule} from "@angular/platform-browser";
import {FormsModule} from "@angular/forms";
import {HttpClientModule} from "@angular/common/http";
import {HttpService} from "./http.service";
import {SharedDataService} from "./shared-data.service";
import {DatasetAnalyzerModule} from "./dataset-analyzer/dataset-analyzer.module";
import {RunAnalyzerModule} from "./run-analyzer/run-analyzer.module";

import { ToastrModule } from 'ngx-toastr';
import {BrowserAnimationsModule} from "@angular/platform-browser/animations";
import {NgApexchartsModule} from "ng-apexcharts";
import { RecommendationsComponent } from './recommendations/recommendations.component';
import { RecommendationInputFieldComponent } from './recommendations/recommendation-input-field/recommendation-input-field.component';

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    RecommendationsComponent,
    RecommendationInputFieldComponent,
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FormsModule,
    HttpClientModule,
    DatasetAnalyzerModule,
    RunAnalyzerModule,
    ToastrModule.forRoot(),
    BrowserAnimationsModule,
    NgApexchartsModule,

  ],
  providers: [HttpService , SharedDataService],
  bootstrap: [AppComponent]
})
export class AppModule { }
