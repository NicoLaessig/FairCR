import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RunInputFieldComponent } from './run-input-field.component';

describe('RunInputFieldComponent', () => {
  let component: RunInputFieldComponent;
  let fixture: ComponentFixture<RunInputFieldComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [RunInputFieldComponent]
    });
    fixture = TestBed.createComponent(RunInputFieldComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
