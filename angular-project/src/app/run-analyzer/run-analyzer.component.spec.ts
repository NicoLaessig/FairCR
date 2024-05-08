import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RunAnalyzerComponent } from './run-analyzer.component';

describe('RunAnalyzerComponent', () => {
  let component: RunAnalyzerComponent;
  let fixture: ComponentFixture<RunAnalyzerComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [RunAnalyzerComponent]
    });
    fixture = TestBed.createComponent(RunAnalyzerComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
