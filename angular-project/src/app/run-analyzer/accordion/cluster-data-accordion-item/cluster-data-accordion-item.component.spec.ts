import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ClusterDataAccordionItemComponent } from './cluster-data-accordion-item.component';

describe('ClusterDataAccordionItemComponent', () => {
  let component: ClusterDataAccordionItemComponent;
  let fixture: ComponentFixture<ClusterDataAccordionItemComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [ClusterDataAccordionItemComponent]
    });
    fixture = TestBed.createComponent(ClusterDataAccordionItemComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
