from django.contrib import admin

# Register your models here.
from .models import RawDataset, Utilisateur,PreprocessedDataset,DataVisualization
class RawDatasetAdmin(admin.ModelAdmin):
    list_display = ('datasetCostumName', 'file_raw_dataset', 'TargetColumn')
    list_filter = ('datasetCostumName', 'file_raw_dataset', 'TargetColumn')
    search_fields = ('datasetCostumName', 'file_raw_dataset', 'TargetColumn')
    ordering = ['datasetCostumName']
class UtilisateurAdmin(admin.ModelAdmin):
    list_display = ('username', 'email', 'profile_picture_path', 'country', 'full_name', 'status')
    list_filter = ('username', 'email', 'profile_picture_path', 'country', 'full_name', 'status')
    search_fields = ('username', 'email', 'profile_picture_path', 'country', 'full_name', 'status')
    ordering = ['username']

admin.site.register(RawDataset, RawDatasetAdmin)
admin.site.register(Utilisateur, UtilisateurAdmin)
admin.site.register(PreprocessedDataset)
admin.site.register(DataVisualization)